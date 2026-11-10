async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(async () => {
    const kernel = window.jupyterapp.shell.currentWidget.sessionContext.session.kernel;
    if (kernel.status !== 'idle') throw new Error('Export requires an idle kernel');
    const future = kernel.requestExecute({code:"import base64,io,zipfile\nfrom pathlib import Path\nexport_buffer=io.BytesIO()\nwith zipfile.ZipFile(export_buffer,'w',zipfile.ZIP_DEFLATED) as export_zip:\n    for export_file in Path('/kaggle/working/cod-reproduction/runs').rglob('*'):\n        if export_file.is_file(): export_zip.write(export_file,export_file.relative_to('/kaggle/working/cod-reproduction'))\nprint(base64.b64encode(export_buffer.getvalue()).decode('ascii'))",silent:false,store_history:false});
    let output='',error='';
    future.onIOPub=message=>{if(message.header.msg_type==='stream')output+=message.content.text;if(message.header.msg_type==='error')error=message.content.ename+': '+message.content.evalue;};
    const reply=await future.done;
    if(reply.content.status!=='ok'||error)throw new Error(error||reply.content.status);
    return output.trim();
  });
}
