// Direct test of sim.proc.print operation only
func.func @entry() {
  %0 = sim.fmt.literal "Hello from arcilator!"
  sim.proc.print %0
  return
}
