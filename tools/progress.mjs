async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(() => {
    const widget=window.jupyterapp.shell.currentWidget;
    const cells=widget.model.toJSON().cells;
    return {status:widget.sessionContext.session?.kernel?.status,cells:cells.map((cell,i)=>({
      i,count:cell.execution_count,tail:(cell.outputs||[]).map(output=>output.text||(output.ename?output.ename+': '+output.evalue:'')).join('').slice(-900)
    })).filter(cell=>cell.i>=2)};
  });
}
