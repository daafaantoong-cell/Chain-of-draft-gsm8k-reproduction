async page => {
  return await page.frame({name:'notebook-editor-cells'}).evaluate(async () => {
    const kernel=window.jupyterapp.shell.currentWidget.sessionContext.session.kernel;
    if(kernel.status!=='idle')throw new Error('Token audit requires an idle kernel');
    const code=`from experiment import get_runtime, load_config, load_prompts, job_prompt
import hashlib
audit_config=load_config(ROOT/'config.json')
audit_runtime=get_runtime(audit_config['model_id'],audit_config['model_revision'],audit_config['dtype'],audit_config['device'])
audit_prompts=load_prompts(ROOT,audit_config['few_shot_count'])
token_audit={'model_revision':audit_config['model_revision'],'tokenizer_class':type(audit_runtime.tokenizer).__name__,'checks':{}}
for audit_phase in ('pilot','full'):
    audit_path=ROOT/'runs'/audit_phase/'results.jsonl'
    audit_bytes=audit_path.read_bytes()
    audit_rows=[json.loads(line) for line in audit_bytes.decode('utf-8').splitlines()]
    for audit_row in audit_rows:
        audit_ids=audit_row['output_token_ids']
        assert len(audit_ids)==audit_row['output_tokens']
        assert audit_runtime.tokenizer.decode(audit_ids,skip_special_tokens=False)==audit_row['raw_output']
        assert audit_runtime.tokenizer.decode(audit_ids,skip_special_tokens=True)==audit_row['output']
        audit_prompt=job_prompt(audit_prompts,audit_row['method'],audit_row['question'],audit_config['few_shot_count'])
        audit_input=audit_runtime.tokenizer.apply_chat_template([{'role':'user','content':audit_prompt}],tokenize=True,add_generation_prompt=True)
        assert len(audit_input)==audit_row['input_tokens']
    token_audit['checks'][audit_phase]={'records':len(audit_rows),'results_sha256':hashlib.sha256(audit_bytes).hexdigest(),'output_decode_and_input_counts':'passed'}
(ROOT/'runs/token_audit.json').write_text(json.dumps(token_audit,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(token_audit,indent=2))`;
    const future=kernel.requestExecute({code,silent:false,store_history:false});
    let output='',error='';
    future.onIOPub=message=>{
      if(message.header.msg_type==='stream')output+=message.content.text;
      if(message.header.msg_type==='error')error=message.content.ename+': '+message.content.evalue;
    };
    const reply=await future.done;
    if(reply.content.status!=='ok'||error)throw new Error(error||reply.content.status);
    return output;
  });
}
