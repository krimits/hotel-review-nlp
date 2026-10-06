import hashlib,json,os,platform,sys,time
from pathlib import Path
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
repo=Path('/workspace/scratch/aa75b9b60470/triage-reliability')
sys.path.insert(0,str(repo/'src'))
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from reviewnlp.triage.qwen_generator import ActionSignals,QwenActionGenerator
from reviewnlp.triage.span_evidence_generator import VERSION,messages_span_evidence,parse_span_issues,prompt_fingerprint
from reviewnlp.triage.demo_service import QWEN_REPO,QWEN_REVISION
root=repo/'docs/experiments/triage_space_publication/e1e37e5/review'
info=json.loads((root/'run.json').read_bytes())
for name,h in info['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
manifest=json.loads((root/'source_manifest.json').read_bytes())
assert prompt_fingerprint()==manifest['runtime_snapshot']['prompt_sha256']
development=json.loads((repo/'docs/experiments/triage_generator/dev.json').read_bytes())
texts={r['id']:r['text'] for r in development['reviews']}
records=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
selected=[r for r in records if r['workflow_error']]
assert [r['id'] for r in selected]==['dev-01','dev-05','dev-08','dev-11']
model_path=Path('/workspace/scratch/aa75b9b60470/qwen-diagnostic-cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots')/QWEN_REVISION
assert (model_path/'model.safetensors').is_file()
torch.set_num_threads(4)
torch.manual_seed(42)
started=time.perf_counter()
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float32,local_files_only=True).eval().to('cpu')
print('Actual pinned Qwen weights loaded in',round(time.perf_counter()-started,1),'seconds; threads',torch.get_num_threads(),flush=True)
report={'kind':'synthetic_failure_reproduction','model':QWEN_REPO,'model_revision':QWEN_REVISION,'real_weights_loaded':True,'workflow':VERSION,'prompt_sha256':prompt_fingerprint(),'source_run_sha256':hashlib.sha256((root/'run.json').read_bytes()).hexdigest(),'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'device':'cpu','dtype':str(model.dtype),'torch_threads':torch.get_num_threads(),'attention_implementation':model.config._attn_implementation,'do_sample':False,'max_new_tokens':400,'precision_matches_hosted':False,'quality_evaluated':False,'reserved_evaluation_used':False,'records':[]}
out=Path('/workspace/scratch/aa75b9b60470/triage_failure_reproduction_cpu.json')
gen=QwenActionGenerator(QWEN_REPO,revision=QWEN_REVISION,loader=lambda:(tokenizer,model),message_builder=messages_span_evidence,prompt_version=VERSION+':issues')
for row in selected:
 signals=ActionSignals(row['sentiment']['label'],row['sentiment']['confidence'],[])
 text=texts[row['id']]
 start=time.perf_counter()
 result=gen.generate(text,signals)
 error=None; parsed=[]
 try:
  if result.hit_token_budget: raise ValueError('hit_token_budget')
  parsed=parse_span_issues(result.raw,text)
 except (ValueError,TypeError) as e: error=str(e)
 r={'id':row['id'],'review':text,'signals':signals.__dict__,'hosted_error':row['workflow_error'],'raw':result.raw,'hit_token_budget':result.hit_token_budget,'parse_error':error,'issues':parsed,'seconds':round(time.perf_counter()-start,3)}
 report['records'].append(r)
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
print('Finished:',out,flush=True)
