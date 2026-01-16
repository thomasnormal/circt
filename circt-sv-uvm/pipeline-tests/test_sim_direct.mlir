// Direct test of sim.proc.print and sim.terminate operations using func.func
// This bypasses the llhd.process issue by using plain functions

func.func @entry() {
  %0 = sim.fmt.literal "Hello!"
  sim.proc.print %0
  sim.terminate success, quiet
  return
}
