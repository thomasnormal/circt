addToLibrary({
  circt_vpi_wasm_yield__sig: "iii",
  circt_vpi_wasm_yield__async: true,
  circt_vpi_wasm_yield: async function(cbFuncPtr, cbDataPtr) {
    const hook =
        typeof globalThis !== "undefined"
            ? globalThis.circtSimVpiYieldHook
            : undefined;
    if (typeof hook === "function")
      await hook();
    else
      await Promise.resolve();
    return await wasmTable.get(cbFuncPtr)(cbDataPtr);
  },
});
