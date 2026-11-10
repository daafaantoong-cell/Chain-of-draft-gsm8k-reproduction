async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(() => {
    const widget=window.jupyterapp.shell.currentWidget;
    const notebook=widget.model.toJSON();
    const original=JSON.stringify(notebook,null,2);
    delete notebook.metadata.kaggle;
    return {files:{
      'debug/jupyter-model-executed.ipynb':original,
      'CoD-Reproduction-executed.ipynb':JSON.stringify(notebook,null,2)
    }};
  });
}
