'use strict';
// Execute dashboard logic against small DOM stubs; this is not a browser preview.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2]||'runs/dashboard/index.html','utf8');
const data=JSON.parse(html.split('<script id="data" type="application/json">')[1].split('</script>')[0]);
const js=fs.readFileSync('web/results/dashboard.js','utf8');
function app(snapshot,code=js){
 const elements=new Map();
 function element(id){if(!elements.has(id))elements.set(id,{value:({study:'main',user:'all',metric:'nonfollowing',search:'','matched-view':'recovery'}[id]||''),textContent:id==='data'?JSON.stringify(snapshot):'',innerHTML:'',hidden:false,disabled:false,open:false,insertAdjacentHTML(_p,v){this.innerHTML+=v;},addEventListener(){},setAttribute(){},showModal(){this.open=true;},close(){this.open=false;}});return elements.get(id);}
 const context=vm.createContext({document:{getElementById:element,addEventListener(){}},location:{hash:'',pathname:'/',search:''},history:{replaceState(){}},setTimeout,URL,Blob,console});
 vm.runInContext(code,context);return {elements,run:s=>vm.runInContext(s,context)};
}
let checks=0;function test(name,fn){fn();console.log('PASS '+name);checks++;}
const ui=app(data);
test('one assignment count, three preserved matched views',()=>{
 assert.equal(data.assigned,145400);assert.equal(data.conditions.length,183);assert.equal(data.groups.length,52);
 const strict=data.conditions.filter(c=>c.family==='MATCHED').flatMap(c=>c.cells.filter(s=>s.user==='all'));
 const recovered=data.conditions.filter(c=>c.family==='MATCHED').flatMap(c=>c.recovery_cells.filter(s=>s.user==='all'));
 assert.equal(strict.reduce((n,c)=>n+c.assigned,0),32256);assert.equal(recovered.reduce((n,c)=>n+c.assigned,0),32256);
 assert.equal(recovered.reduce((n,c)=>n+c.unknown,0),0);
 const prior=data.conditions.filter(c=>c.family==='MATCHED').flatMap(c=>c.prior_recovery_cells.filter(s=>s.user==='all'));assert.equal(prior.reduce((n,c)=>n+c.unknown,0),1);
 assert.equal(recovered.reduce((n,c)=>n+(c.labels.no_implementation||0),0),1);
 const syntax=data.conditions.filter(c=>c.family.startsWith('MATCHED_')).flatMap(c=>c.recovery_cells.filter(s=>s.user==='all'));assert.equal(syntax.reduce((n,c)=>n+c.assigned,0),20736);assert.equal(syntax.reduce((n,c)=>n+c.unknown,0),3456);assert.ok(strict.reduce((n,c)=>n+c.unknown,0)>1);
});
test('view switch selects matching cells, inference, and CSV source',()=>{
 ui.elements.get('study').value='MATCHED';
 for(const view of ['strict','prior','recovery']){
  ui.elements.get('matched-view').value=view;ui.run('render()');
  assert.equal(ui.run('pairedExportRows.length'),30);
  const source=view==='strict'?data.matched_review:(view==='prior'?data.matched_prior_recovery:data.matched_recovery);
  for(const user of ['all','loops','comprehensions']){
   ui.elements.get('user').value=user;ui.run('render()');
   const exports=JSON.parse(ui.run('JSON.stringify(pairedExportRows)'));let i=0;
   for(const family of source.analysis.families)for(const comparison of family.comparisons){
    const row=exports[i++],v=user==='all'?comparison:comparison.by_user_preference.find(x=>x.user_preference===user);
    assert.equal(row.source,source.analysis_source);assert.equal(row.view.estimate,v.estimate);assert.deepEqual(row.view.identification_bounds,v.identification_bounds);assert.deepEqual(row.view.interval95,v.interval95);assert.equal(row.holm,user==='all'?comparison.holm_p_value:null);
   }
   assert.ok(ui.run('pairedCSV()').includes(source.analysis_source));
  }
  ui.elements.get('user').value='all';
 }
});
test('syntax panels retain whole cross-style families and held bounds',()=>{
 ui.elements.get('study').value='MATCHED_NAMING';ui.run('render()');
 assert.equal(ui.run('pairedExportRows.length'),24);
 assert.deepEqual(JSON.parse(ui.run('JSON.stringify([...new Set(pairedExportRows.map(r=>r.family_size))])')),[6,12]);
 const c=data.conditions.find(c=>c.family==='MATCHED_NAMING'&&c.relation==='absent');
 const value=ui.run(`counts(conditions.get(${JSON.stringify(c.id)}),'glm')`);
 assert.equal(value.rate,null);assert.equal(value.denominator,384);assert.equal(value.lower,0);assert.equal(value.upper,1);assert.equal(value.ci,null);
 assert.ok(ui.elements.get('charts').innerHTML.includes('withheld unresolved'));
 assert.ok(ui.elements.get('charts').innerHTML.includes('No evidence / assigned'));
 assert.equal(ui.elements.get('metric').disabled,true);
});
test('syntax directional exports retain full family sizes and never invent p values',()=>{
 for(const [study,users] of [['MATCHED_NAMING',['snake_case','camelCase']],['MATCHED_QUOTES',['single_quotes','double_quotes']]]){
  ui.elements.get('study').value=study;
  for(const user of users){ui.elements.get('user').value=user;ui.run('render()');const out=JSON.parse(ui.run('JSON.stringify(pairedExportRows)'));assert.equal(out.length,12);assert.ok(out.every(x=>x.p===null&&x.holm===null));assert.ok(ui.run('pairedCSV()').includes(data.matched_syntax_recovery.analysis_source));}
 }
});
test('new prompts retain exact saved styles and reviewer presence',()=>{
 for(const c of data.conditions.filter(c=>c.family.startsWith('MATCHED_'))){ui.run(`showPrompt(${JSON.stringify(c.id)},false)`);assert.ok(ui.elements.get('prompt-text').textContent);assert.ok(ui.elements.get('prompt-description').textContent.includes(c.relation==='absent'?'reviewer not mentioned':'reviewer'));}
});
if(process.argv[3])test('historical cells, prompts, panels and original strict exports unchanged',()=>{
 const before=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
 assert.equal(before.assigned,145400);
 const current=new Map(data.conditions.map(c=>[c.id,c]));
 for(const old of before.conditions){const now={...current.get(old.id)};delete now.prior_recovery_cells;delete now.recovery_cells;const expected={...old};delete expected.recovery_cells;assert.deepEqual(now,expected);if(old.recovery_cells)assert.deepEqual(current.get(old.id).prior_recovery_cells,old.recovery_cells);}
 assert.deepEqual(data.groups.slice(0,before.groups.length),before.groups);
 assert.deepEqual(data.sources.filter(s=>before.sources.some(old=>old.path===s.path)),before.sources);
 assert.deepEqual(data.matched_review,before.matched_review);assert.deepEqual(data.matched_syntax,before.matched_syntax);assert.deepEqual(data.matched_prior_recovery,before.matched_recovery);
});
console.log(`${checks} dashboard logic checks passed`);
