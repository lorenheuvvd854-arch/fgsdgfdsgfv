THEOS ?= /opt/theos
LDID ?= /usr/local/bin/ldid
TARGET := iphone:clang:latest:14.0
ARCHS := arm64

include $(THEOS)/makefiles/common.mk

APPLICATION_NAME := F1OOBPoC
F1OOBPoC_FILES := main.m cg.m jb.m
F1OOBPoC_RESOURCE_FILES := Info.plist
F1OOBPoC_FRAMEWORKS := UIKit Foundation IOKit IOSurface Metal CoreGraphics CoreFoundation VideoToolbox CoreMedia CoreVideo QuickLookThumbnailing WebKit
F1OOBPoC_CFLAGS := -fobjc-arc -O2 -Wall -Wno-unused-variable -Wno-unused-function
F1OOBPoC_LDFLAGS := -fuse-ld=/usr/bin/ld64.lld -Wl,-platform_version,ios,14.0,14.0
F1OOBPoC_CODESIGN_FLAGS := -Sentitlements.plist

include $(THEOS)/makefiles/application.mk

.PHONY: ipa
ipa: all
	mkdir -p .theos/ipa/Payload
	rm -rf .theos/ipa/Payload/F1OOBPoC.app
	cp -a .theos/obj/debug/F1OOBPoC.app .theos/ipa/Payload/
	$(LDID) -Sentitlements.plist .theos/ipa/Payload/F1OOBPoC.app/F1OOBPoC
	cd .theos/ipa && rm -f ../../F1OOB-PoC.ipa && zip -qry ../../F1OOB-PoC.ipa Payload
