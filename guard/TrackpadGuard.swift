// trackpad-guard: stops the trackpad from moving the cursor, clicking, scrolling or triggering gestures while armed.
//
// The raw multitouch signal the recorder reads comes from a lower layer (MultitouchSupport), so recording keeps
// working while the pointer is frozen. Controlled over stdin by the Python server, one command per line:
//     block | allow | ping | quit
// Prints to stdout: ready | blocked | allowed | escape | watchdog | error: ...
//
// Safety: Esc always releases the block. If the server stops pinging for 3 s, the block is released. If the server
// exits (stdin closes), this process exits and macOS removes the event tap.
//
// Build: swiftc -O guard/TrackpadGuard.swift -o bin/trackpad-guard

import ApplicationServices
import Foundation

let escapeKeyCode: Int64 = 53
let watchdogSeconds = 3.0

var blocking = false
var lastPing = Date()
var tap: CFMachPort?

func say(_ line: String) {
    print(line)
    fflush(stdout)
}

// Pointer and gesture event types to swallow. 29-37 are NSEvent gesture types without CGEventType names.
let blockedTypes: [UInt32] = [
    CGEventType.leftMouseDown.rawValue, CGEventType.leftMouseUp.rawValue,
    CGEventType.rightMouseDown.rawValue, CGEventType.rightMouseUp.rawValue,
    CGEventType.mouseMoved.rawValue, CGEventType.leftMouseDragged.rawValue, CGEventType.rightMouseDragged.rawValue,
    CGEventType.scrollWheel.rawValue,
    CGEventType.otherMouseDown.rawValue, CGEventType.otherMouseUp.rawValue, CGEventType.otherMouseDragged.rawValue,
    18, 19, 20,                  // rotate, begin gesture, end gesture
    29, 30, 31, 32, 33, 34, 37,  // gesture, magnify, swipe, smart magnify, quick look, pressure, direct touch
]

let callback: CGEventTapCallBack = { _, type, event, _ in
    // macOS switches a tap off if it is ever slow to respond; switch it straight back on.
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        if let tap = tap { CGEvent.tapEnable(tap: tap, enable: true) }
        return Unmanaged.passUnretained(event)
    }
    if type == .keyDown {
        if blocking && event.getIntegerValueField(.keyboardEventKeycode) == escapeKeyCode {
            blocking = false
            say("escape")
        }
        return Unmanaged.passUnretained(event)  // keys always pass through
    }
    return blocking ? nil : Unmanaged.passUnretained(event)
}

if CommandLine.arguments.contains("--check") {
    say(AXIsProcessTrusted() ? "trusted" : "untrusted")
    exit(0)
}

// Event taps that can drop events need Accessibility permission. Ask macOS to show its prompt if missing.
let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
if !AXIsProcessTrustedWithOptions(options) {
    say("error: accessibility permission needed")
    exit(3)
}

var mask: CGEventMask = CGEventMask(1) << CGEventMask(CGEventType.keyDown.rawValue)
for t in blockedTypes { mask |= CGEventMask(1) << CGEventMask(t) }

guard let created = CGEvent.tapCreate(tap: .cghidEventTap, place: .headInsertEventTap, options: .defaultTap,
                                      eventsOfInterest: mask, callback: callback, userInfo: nil) else {
    say("error: could not create event tap")
    exit(4)
}
tap = created
CFRunLoopAddSource(CFRunLoopGetCurrent(), CFMachPortCreateRunLoopSource(nil, created, 0), .commonModes)
CGEvent.tapEnable(tap: created, enable: true)

// Commands from the server, read on a background thread and applied on the main run loop.
Thread.detachNewThread {
    while let line = readLine() {
        let cmd = line.trimmingCharacters(in: .whitespaces)
        DispatchQueue.main.async {
            switch cmd {
            case "block": blocking = true; lastPing = Date(); say("blocked")
            case "allow": blocking = false; say("allowed")
            case "ping": lastPing = Date()
            case "quit": exit(0)
            default: say("error: unknown command \(cmd)")
            }
        }
    }
    exit(0)  // stdin closed: the server is gone, so never leave the trackpad blocked
}

Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
    if blocking && Date().timeIntervalSince(lastPing) > watchdogSeconds {
        blocking = false
        say("watchdog")
    }
}

say("ready")
CFRunLoopRun()
