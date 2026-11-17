"""Previous-day 18Z HRRR runs (and the ASNOW lead axis of the 00Z runs) as county-polygon statistics.

    python data_pipeline/hrrr/runs_18z.py full --root <work/hrrr_18z> --census <tl_2025_us_county.zip> --start 20250531 --end 20250830
    python data_pipeline/hrrr/runs_18z.py coordinate00 --root <work/hrrr_18z> --start 20250530 --end 20250831

full: for each date, the 18Z run's objects are fetched from s3://hrrrzarr (anonymous; raw bytes and version receipts are
kept under <root>/objects, so a rerun reads the cache) and reduced to mean / std / max / p95 per county and target hour
-> <root>/daily/<YYYYMMDD>_18z.parquet. coordinate00: the verified time coordinate of each 00Z run -> <root>/coordinates00
(input of snow_axis.py). Levels, chunks and the county-to-cell plan are taken from data_pipeline/reference/hrrr_grid.
"""
import sys,json,hashlib,ctypes,time,argparse,threading,os,shutil
from pathlib import Path
from datetime import datetime,timedelta,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,polars as pl,boto3
from botocore import UNSIGNED
from botocore.config import Config
PIPELINE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PIPELINE))
from hrrr.polygons import load_county_geometries,build_spatial_plan
from hrrr.decode import _library,decode_chunk_with_metadata
from hrrr.stats import robust_spatial_stats
REFERENCE=PIPELINE/'reference'/'hrrr_grid'
COUNTIES=(PIPELINE/'reference'/'counties_40.txt').read_text().strip().split(',')
VARIABLES=['TMP','DPT','UGRD','VGRD','GUST','CAPE','RH','PRES','MSLMA','TCDC','VIS','PRATE','APCP_1hr_acc_fcst','ASNOW_acc_fcst']
STATS=['mean','std','max','p95']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
RAW=None;CENSUS=None;UTC=timezone.utc;WINDOW=None  # RAW / CENSUS are set from the command line
LONG_SCHEMA={'fips_code':pl.String,'model_run_utc':pl.Int64,'valid_time_utc':pl.Int64,'forecast_hour':pl.Int64,'variable':pl.String,'units':pl.String,'availability':pl.Float64,'quality_status':pl.String,'valid_area_ratio':pl.Float64,**{s:pl.Float64 for s in STATS}}
CLIENT=boto3.client('s3',region_name='us-east-1',config=Config(signature_version=UNSIGNED,max_pool_connections=384,retries={'max_attempts':3},connect_timeout=10,read_timeout=45))
LEVELS={'TMP':'2m_above_ground','DPT':'2m_above_ground','RH':'2m_above_ground','UGRD':'10m_above_ground','VGRD':'10m_above_ground'}
# Other levels are read from the retained audited source, never guessed.
def setup():
 RAW.mkdir(parents=True,exist_ok=True)
 for name in ('configuration.json','spatial_plan.parquet','polygon_coverage.json'):  # shipped levels, chunks and cell plan
  if not (RAW/name).exists():shutil.copyfile(REFERENCE/name,RAW/name)
 if (RAW/'configuration.json').exists():  # data_pipeline: levels and chunks shipped with the package
  shipped=json.loads((RAW/'configuration.json').read_text());levels,chunks=shipped['levels'],shipped['chunks']
 else:
  raise SystemExit('configuration.json (levels and chunks) is missing from data_pipeline/reference/hrrr_grid')
 p=RAW/'spatial_plan.parquet'
 if not p.exists():
  census=CENSUS
  rows,coverage=build_spatial_plan(load_county_geometries(census,COUNTIES),chunks)
  pl.DataFrame(rows).write_parquet(p);dump(RAW/'polygon_coverage.json',{'census_sha256':sha(census),'coverage':coverage,'chunks':chunks})
 plan=pl.read_parquet(p);county={}
 for c in COUNTIES:
  f=plan.filter(pl.col('fips_code')==c).sort('chunk_id','flat_index');county[c]=(f['chunk_id'].to_numpy(),f['flat_index'].to_numpy(),f['intersection_area_m2'].to_numpy(),f)
 dump(RAW/'configuration.json',{'levels':levels,'chunks':chunks,'variables':VARIABLES,'stats':STATS,'version_policy':'raw availability stores GET response LastModified for provenance; replay cutoff chosen separately; update time is not first publication','time_policy':'actual time coordinate; missing native times not interpolated','units_policy':'preserve attrs; unknown remains unknown; no invented precipitation/snow units','spatial_plan_sha256':sha(p)})
 return levels,chunks,county

def get(key):
 p=RAW/'objects'/key;r=p.with_name(p.name+'.receipt.json')
 if p.exists() and r.exists():
  try:
   receipt=json.loads(r.read_text())
   if receipt.get('status')=='OK' and receipt.get('bytes')==p.stat().st_size:
    release_raw_file_cache(r);return p,receipt
  except (ValueError,OSError):pass
 p.parent.mkdir(parents=True,exist_ok=True)
 try:
  o=CLIENT.get_object(Bucket='hrrrzarr',Key=key);b=o['Body'].read()
  receipt={'key':key,'status':'OK','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'last_modified':o['LastModified'].isoformat(),'etag':o.get('ETag'),'retrieved_at':datetime.now(UTC).isoformat()}
  p.write_bytes(b);dump(r,receipt);release_raw_file_cache(r);return p,receipt
 except Exception as e:
  receipt={'key':key,'status':'MISSING_OR_FAILED','error':str(e),'retrieved_at':datetime.now(UTC).isoformat()};dump(r,receipt);return None,receipt

def decode_coord(p,meta):
 b=p.read_bytes()
 if meta.get('compressor'):
  buf=ctypes.create_string_buffer(100000);n=_library().blosc_decompress(b,buf,100000)
  if n<=0:raise ValueError('coordinate decompression failed')
  b=buf.raw[:n]
 return np.frombuffer(b,dtype=meta['dtype'])[:meta['shape'][0]].copy()

def release_raw_file_cache(path):
 # Only this task's raw objects. Preserve bytes/mtime and avoid global cache settings.
 if hasattr(os,'posix_fadvise'):
  with path.open('rb') as handle:
   os.fdatasync(handle.fileno())
   os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)

def stats(x,w):
 valid=np.isfinite(x)&(x!=-10000)&(x!=-9999);a=valid*w;total=a.sum(axis=1);coverage=total/w.sum()
 safe=np.where(valid,x,0);mean=(a*safe).sum(axis=1)/np.maximum(total,1e-100)
 std=np.sqrt((a*np.where(valid,x-mean[:,None],0)**2).sum(axis=1)/np.maximum(total,1e-100))
 maximum=np.where(valid,x,-np.inf).max(axis=1)
 order=np.argsort(np.where(valid,x,np.inf),axis=1,kind='stable');sx=np.take_along_axis(x,order,axis=1);sw=np.take_along_axis(a,order,axis=1)
 qi=(np.cumsum(sw,axis=1)>=.95*total[:,None]).argmax(axis=1);p95=sx[np.arange(len(x)),qi]
 out=np.column_stack([mean,std,maximum,p95]);out[coverage<.99]=np.nan
 return out,coverage

def day(date,hour,levels,chunks,county,coord_only=False):
 started=time.perf_counter()
 tag=f'{date}_{hour:02}z';dst=RAW/'daily'/f'{tag}.parquet';receipt_path=RAW/'daily'/f'{tag}.json'
 if dst.exists() and receipt_path.exists() and not coord_only:
  result=json.loads(receipt_path.read_text());return {k:result[k] for k in ['tag','rows','version_time_before_next00']}
 run=int(datetime.strptime(date,'%Y%m%d').replace(hour=hour,tzinfo=UTC).timestamp());rows=[];receipts=[];audits=[]
 variables=['ASNOW_acc_fcst'] if coord_only else VARIABLES
 variable_keys={}
 for v in variables:
  level=levels[v];prefix=f'sfc/{date}/{tag}_fcst.zarr/{level}/{v}'
  coord='time_2' if v=='ASNOW_acc_fcst' else 'time';cp=f'{prefix}/{coord}'
  keys=[prefix+'/.zmetadata',cp+'/0']
  if not coord_only:keys += [f'{prefix}/{level}/{v}/{c}' for c in chunks]
  variable_keys[v]=keys
 # One bounded pool for this day: receipts/path handles only, no retained
 # decoded arrays. Avoid fourteen independent slow-tail waits and pool churn.
 all_keys=[key for keys in variable_keys.values() for key in keys]
 with ThreadPoolExecutor(max_workers=12) as pool:
  fetched=dict(zip(all_keys,pool.map(get,all_keys)))
 for v in variables:
  level=levels[v];prefix=f'sfc/{date}/{tag}_fcst.zarr/{level}/{v}'
  coord='time_2' if v=='ASNOW_acc_fcst' else 'time';cp=f'{prefix}/{coord}'
  results=[fetched[key] for key in variable_keys[v]]
  receipts.extend(r for _,r in results)
  if not results[0][0] or not results[1][0]:
   audits.append({'variable':v,'status':'NO_METADATA_OR_COORD'});continue
  metadata=json.loads(results[0][0].read_text())['metadata'];spec=metadata[f'{level}/{v}/.zarray'];attrs=metadata[f'{level}/{v}/.zattrs'];cm=metadata[coord+'/.zarray'];ca=metadata[coord+'/.zattrs']
  if ca.get('units')!='hours since 1970-01-01':raise ValueError(f'Unsupported actual time units {ca}')
  target=np.rint(decode_coord(results[1][0],cm)*3600).astype(np.int64);lead=(target-run)/3600
  audit={'variable':v,'units':attrs.get('units','unknown'),'cell_methods':attrs.get('cell_methods'),'shape':spec['shape'],'dtype':spec['dtype'],'actual_leads':lead.tolist(),'coordinate_units':ca.get('units'),'status':'COORDINATE_VERIFIED'};audits.append(audit)
  if coord_only:continue
  arrays={};decode_errors=[]
  for c,(p,r) in zip(chunks,results[2:]):
   if not p:continue
   try:
    ar,dm=decode_chunk_with_metadata(p,variable=v,dtype_hint=spec['dtype'])
    if len(ar)!=len(target):raise ValueError('time/data shape mismatch')
    arrays[c]=ar.reshape(len(ar),-1)
   except Exception as e:decode_errors.append({'chunk':c,'error':str(e)})
   finally:release_raw_file_cache(p)
  audit['decode_errors']=decode_errors
  # Match retained materializer's conservative union-of-required-tiles availability.
  available=all(r['status']=='OK' for _,r in results) and len(arrays)==len(chunks)
  availability=max(datetime.fromisoformat(r['last_modified']).timestamp() for _,r in results) if available else None
  for c,(cs,ix,w,f) in county.items():
   x=np.full((len(target),len(ix)),np.nan)
   for chunk in set(cs):
    mask=cs==chunk
    if chunk in arrays:x[:,mask]=arrays[chunk][:,ix[mask]]
   out,coverage=stats(x,w)
   # One sample row per variable is checked against the established full-stat implementation.
   check=robust_spatial_stats(x[0],w,center_inside_polygon=f['center_inside_polygon'].to_numpy(),cell_overlap_fraction=f['cell_overlap_fraction'].to_numpy(),boundary_cell=f['boundary_cell'].to_numpy(),valid_mask=np.isfinite(x[0])&(x[0]!=spec['fill_value']))
   expected=[check[k] for k in ['area_mean','area_std','raw_max','weighted_p95']]
   if coverage[0]>=.99:np.testing.assert_allclose(out[0],expected,rtol=1e-10,atol=1e-10)
   for i,t in enumerate(target):rows.append({'fips_code':c,'model_run_utc':run,'valid_time_utc':int(t),'forecast_hour':int(lead[i]),'variable':v,'units':attrs.get('units','unknown'),'availability':availability,'quality_status':'VALID' if coverage[i]>=.99 else 'INSUFFICIENT_VALID_AREA','valid_area_ratio':float(coverage[i]),**{s:float(out[i,j]) for j,s in enumerate(STATS)}})
 result={'tag':tag,'run':run,'rows':len(rows),'objects':receipts,'audit':audits,'version_time_before_next00':all(r['status']=='OK' and datetime.fromisoformat(r['last_modified']).timestamp()<=run+(24-hour)*3600 for r in receipts),'runtime_seconds':time.perf_counter()-started}
 if coord_only:
  dump(RAW/'coordinates00'/f'{tag}.json',result)
 else:
  dst.parent.mkdir(exist_ok=True);pl.DataFrame(rows,schema=LONG_SCHEMA).write_parquet(dst);dump(receipt_path,result)
 return {k:result[k] for k in ['tag','rows','version_time_before_next00']}

def dates():
 if WINDOW:  # data_pipeline: --start/--end (YYYYMMDD) limit the download to one window
  d=datetime.strptime(WINDOW[0],'%Y%m%d');end=datetime.strptime(WINDOW[1],'%Y%m%d')
  while d<=end:yield d.strftime('%Y%m%d');d+=timedelta(days=1)
  return
 d=datetime(2021,12,31);end=datetime(2024,12,31)
 while d<=end:yield d.strftime('%Y%m%d');d+=timedelta(days=1)
 d=datetime(2025,5,30);end=datetime(2025,8,31)
 while d<=end:yield d.strftime('%Y%m%d');d+=timedelta(days=1)
def scheduled_day(date,levels,chunks,county):
 pause=RAW/'acquisition.pause';active=RAW/'active_downloads'/f'{os.getpid()}_{date}.json'
 while pause.exists():time.sleep(1)
 active.parent.mkdir(exist_ok=True);dump(active,{'pid':os.getpid(),'date':date,'started':time.time()})
 try:return day(date,18,levels,chunks,county)
 finally:active.unlink(missing_ok=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['sample','full','coordinate00']);p.add_argument('--root',type=Path,required=True);p.add_argument('--census',type=Path);p.add_argument('--start');p.add_argument('--end');a=p.parse_args()
 RAW,CENSUS=a.root.resolve(),a.census
 if a.start:WINDOW=(a.start,a.end or a.start)
 levels,chunks,county=setup()
 print(json.dumps({'chunks':chunks,'levels':levels}),flush=True)
 if a.stage=='sample':
  for date in ['20211231','20220510','20220928','20230510','20230824','20240510','20240708']:
   print(json.dumps(day(date,18,levels,chunks,county)),flush=True)
  print(json.dumps(day('20230510',0,levels,chunks,county)),flush=True)
 elif a.stage=='full':
  dump(RAW/'acquisition_process.json',{'pid':os.getpid(),'day_workers':4,'object_workers_per_day':12,'checkpoint':'completed daily parquet + version receipts; raw cache resumes partial days'})
  with ThreadPoolExecutor(max_workers=4) as pool:
   pending=[pool.submit(scheduled_day,d,levels,chunks,county) for d in dates()]
   for n,f in enumerate(as_completed(pending),1):print(json.dumps({'done':n,'total':len(pending),**f.result()}),flush=True)
  if not WINDOW:print(json.dumps(day('20211231',0,levels,chunks,county)),flush=True)  # 2022 boundary run
 else:
  with ThreadPoolExecutor(max_workers=12) as pool:
   pending=[pool.submit(day,d,0,levels,chunks,county,True) for d in dates()]
   for n,f in enumerate(as_completed(pending),1):
    if n%50==0 or n==len(pending):print(json.dumps({'done':n,'total':len(pending),'last':f.result()}),flush=True)
