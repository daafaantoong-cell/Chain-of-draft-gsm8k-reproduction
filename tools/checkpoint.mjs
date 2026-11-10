async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(async () => {
    const manager=window.jupyterapp.serviceManager.contents;
    const files={};
    /** Read a directory and copy each UTF-8 result asset through Jupyter Contents. */
    async function readDirectory(path) {
      const listing=await manager.get(path,{content:true,type:'directory'});
      for(const item of listing.content) {
        if(item.type==='directory') await readDirectory(item.path);
        else {
          const file=await manager.get(item.path,{content:true,format:'text'});
          files[item.path.replace('cod-reproduction/','')]=file.content;
        }
      }
    }
    await readDirectory('cod-reproduction/runs');
    return {files};
  });
}
