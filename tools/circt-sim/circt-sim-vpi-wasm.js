addToLibrary({
  circt_vpi_wasm_yield__sig: "iii",
  circt_vpi_wasm_yield: function(cbFuncPtr, cbDataPtr) {
    return Asyncify.handleAsync(async function() {
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
        return wasmTable.get(cbFuncPtr)(cbDataPtr);
      return 0;
    });
  },
});
