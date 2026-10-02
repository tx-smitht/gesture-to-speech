// Two faces on one connection: the simple guided flow (the default) and the full advanced app.
import { AdvancedApp } from "./AdvancedApp";
import { useServer } from "./server";
import { usePersistent } from "./simple/persist";
import { SimpleApp } from "./simple/SimpleApp";

export default function App() {
  const server = useServer();
  const [advanced, setAdvanced] = usePersistent("trackpad.advanced", false);
  return advanced
    ? <AdvancedApp server={server} onSimple={() => setAdvanced(false)} />
    : <SimpleApp server={server} onAdvanced={() => setAdvanced(true)} />;
}
