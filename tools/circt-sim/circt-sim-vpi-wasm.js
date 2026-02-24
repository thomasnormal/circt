addToLibrary({
  circt_vpi_wasm_yield__sig: "iii",
  circt_vpi_wasm_yield__async: true,
  circt_vpi_wasm_yield: async function(cbFuncPtr, cbDataPtr) {
    const hook =
        typeof globalThis !== "undefined"
            ? globalThis.circtSimVpiYieldHook
            : undefined;
    if (typeof hook === "function")
      await hook(cbFuncPtr, cbDataPtr);
    else
      await Promise.resolve();
    // cbFuncPtr may be 0 when the JS caller uses cbRtn=0 and relies entirely
    // on the yield hook for dispatch (avoids needing Emscripten addFunction).
    if (cbFuncPtr)
      return await wasmTable.get(cbFuncPtr)(cbDataPtr);
    return 0;
  },
});
