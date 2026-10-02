// speaker: keeps a speech voice loaded and says each line from stdin the moment it arrives.
//
// Starting `say` for every word costs about a second: the process starts and loads the voice before any sound.
// This helper loads the voice once, so a word starts within a few tens of milliseconds. Used by speech.py.
//
// stdin, one command per line:   <text to say>   |   quit
// stdout:                        ready   |   started <ms from receiving the line to the first sound>
//
// Build: swiftc -O guard/Speaker.swift -o bin/speaker

import AVFoundation
import Foundation

final class Speaker: NSObject, AVSpeechSynthesizerDelegate {
    let synth = AVSpeechSynthesizer()
    var received: [ObjectIdentifier: Date] = [:]

    override init() {
        super.init()
        synth.delegate = self
    }

    func say(_ text: String) {
        let u = AVSpeechUtterance(string: text)
        u.preUtteranceDelay = 0
        u.postUtteranceDelay = 0
        received[ObjectIdentifier(u)] = Date()
        synth.speak(u)  // queued: words are spoken in order, each as soon as the previous one ends
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didStart u: AVSpeechUtterance) {
        let ms = received.removeValue(forKey: ObjectIdentifier(u)).map { Date().timeIntervalSince($0) * 1000 } ?? -1
        print(String(format: "started %.0f", ms))
        fflush(stdout)
    }
}

let speaker = Speaker()
speaker.say(" ")  // warm up: loads the voice and opens the audio device now, not on the first real word
print("ready")
fflush(stdout)

DispatchQueue.global().async {
    while let line = readLine() {
        let text = line.trimmingCharacters(in: .whitespaces)
        if text == "quit" { exit(0) }
        if !text.isEmpty { DispatchQueue.main.async { speaker.say(text) } }
    }
    exit(0)  // stdin closed: the Python side has gone
}
RunLoop.main.run()
