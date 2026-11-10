async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(async () => {
    const kernel = window.jupyterapp.shell.currentWidget.sessionContext.session.kernel;
    if (kernel.status !== 'idle') throw new Error('Runtime smoke requires an idle kernel');
    const code = "from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM\nimport transformers\nconfig_smoke=AutoConfig.from_pretrained('Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')\nassert config_smoke._commit_hash=='aa8e72537993ba99e69dfaafa59ed015b17504d1'\nprint('Runtime smoke passed',transformers.__version__,config_smoke._commit_hash)\nold_pilot=ROOT/'runs/pilot'\nif old_pilot.exists() and (old_pilot/'failure.json').exists():\n    (ROOT/'debug').mkdir(exist_ok=True)\n    old_pilot.rename(ROOT/'debug/transformers5-first-attempt')";
    const future = kernel.requestExecute({code,silent:false,store_history:false});
    let output='', error='';
    future.onIOPub=message=>{
      if(message.header.msg_type==='stream')output+=message.content.text;
      if(message.header.msg_type==='error')error=message.content.ename+': '+message.content.evalue;
    };
    const reply=await future.done;
    return {status:reply.content.status,output,error};
  });
}
