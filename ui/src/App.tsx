// Two faces on one connection: the simple guided flow (the default) and the full advanced app.
import { AdvancedApp } from "./AdvancedApp";
import { useServer } from "./server";
import { usePersistent } from "./simple/persist";
import { SimpleApp } from "./simple/SimpleApp";

export default function App() {
  const server = useServer();
  const [advanced, setAdvanced] = usePersistent("trackpad.advanced", false);

  // Start over: a new, untrained decoder, then the simple view's intro from the beginning (?intro resets it)
  const startOver = (freshRecordings: boolean) => {
    server.send("decoder_new", { fresh_recordings: freshRecordings });
    history.replaceState(null, "", `${location.pathname}?intro`);
    setAdvanced(false);
  };

  return advanced
    ? <AdvancedApp server={server} onSimple={() => setAdvanced(false)} onStartOver={startOver} />
    : <SimpleApp server={server} onAdvanced={() => setAdvanced(true)} />;
}
