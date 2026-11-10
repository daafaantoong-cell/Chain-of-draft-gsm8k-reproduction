async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(() => {
    const w = window.jupyterapp.shell.currentWidget;
    return {status:w.sessionContext.session?.kernel?.status, cells:w.model.toJSON().cells.map((c,i)=>({
      i, count:c.execution_count, output:(c.outputs||[]).map(o=>o.text||(o.ename?o.ename+': '+o.evalue:'')).join('').slice(-1500)
    }))};
  });
}
