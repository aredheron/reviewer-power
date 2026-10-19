'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
const conditions = new Map(DATA.conditions.map(c => [c.id, c]));
const modelOrder = Object.keys(DATA.models);
const modelColors = Object.fromEntries(modelOrder.map((m,i)=>[m,['#386cb0','#d35e38','#25876a','#9363b5','#b68a14','#c44e85','#338eaa','#73614c','#506580'][i]]));
const fmt = n => n.toLocaleString();
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = n => n === null ? '—' : (100*n).toFixed(1)+'%';
const money = n => n.toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2});
let currentCondition, currentPrompt = 0, templateMode = true;
let exportRows = [];
let pairedExportRows = [];
const isMatched = c => ['MATCHED','MATCHED_NAMING','MATCHED_QUOTES'].includes(c.family);
const recoverySelected = () => DATA.matched_recovery && $('matched-view').value === 'recovery';
const signedPercent = n => n == null ? '—' : (n > 0 ? '+' : '')+(100*n).toFixed(2)+' pp';
const signedRange = values => values ? values.map(signedPercent).join(' to ') : '—';
const armName = arm => ({absent:'No reviewer mentioned',reviewer_unseen:'Reviewer will not see output'}[arm] || arm.replace(/_no_consequences$/, ' · no consequences').replace(/_rl_update$/, ' · RL updates').replace(/_future_selection$/, ' · future selection'));
const familyNames = {
  primary_no_consequences_vs_absence:'Primary · No consequences minus absence',
  secondary_consequences:'Secondary · Consequences minus no consequences',
  secondary_unseen_vs_absence:'Secondary · Unseen output minus absence'
};

function renderStudyStatus() {
  const notices=DATA.status_notices||[];
  $('study-status').hidden=!notices.length;
  $('study-status').innerHTML=notices.map(n=>{
    const main=n.kind==='main_progress', c=n.main_counts;
    const body=main
      ? `<p>${fmt(c.started)} of ${fmt(c.assigned)} main assignments started · ${fmt(c.saved)} saved · ${fmt(c.scored)} scored · ${fmt(c.resolved)} resolved labels · ${fmt(c.terminal_unresolved)} terminal unresolved · ${fmt(c.pending)} pending judgments.</p><p>${esc(n.reason)}</p><p><strong>Whole-study recorded spend ${money(n.whole_study_recorded_usd)}</strong> · Outstanding reservations ${money(n.outstanding_usd)} · Hard ceiling ${money(n.cap_usd)}.</p>`
      : `<p>${fmt(n.main_started)} of ${fmt(n.main_assigned)} planned main assignments started. Naming and quote main collection are on hold.</p><ul>${n.hold_reasons.filter(reason=>!reason.toLowerCase().includes('forecast')).map(reason=>`<li>${esc(reason)}</li>`).join('')}</ul><p><strong>Recorded validation spend ${money(n.validation_recorded_usd)}</strong> · Whole-study recorded spend ${money(n.whole_study_recorded_usd)} · Outstanding reservations ${money(n.outstanding_usd)}.</p><p>Whole-study forecast ${money(n.forecast_usd)} with ${percent(n.forecast_main_margin)} main-cost margin, against the ${money(n.cap_usd)} ceiling.</p>`;
    return `<section class="study-notice"><span class="eyebrow">${main?'MAIN':'VALIDATION'} SNAPSHOT · ${esc(new Date(n.as_of).toLocaleString())}</span><h2>${esc(n.title)}</h2>${body}<details><summary>Audit sources</summary>${n.sources.map(s=>`<p class="mono">${esc(s.path)}<br>SHA-256 ${esc(s.sha256)}</p>`).join('')}</details></section>`;
  }).join('');
}
renderStudyStatus();

function counts(condition, model) {
  const available = $('matched-view').value === 'prior' && condition.prior_recovery_cells ? condition.prior_recovery_cells : (recoverySelected() && condition.recovery_cells ? condition.recovery_cells : condition.cells);
  const cell = available.find(c => c.model === model && c.user === $('user').value);
  if (!cell) return null;
  const result = {...cell};
  result.ci = cell.interval95[$('metric').value];
  result.numerator = result[$('metric').value];
  result.assessed = cell.assessed ?? cell.resolved;
  result.no_evidence = cell.no_evidence ?? 0;
  result.mixed = cell.mixed ?? cell.labels.mixed ?? 0;
  result.neither = cell.labels.neither ?? 0;
  result.no_implementation = cell.labels.no_implementation ?? 0;
  result.invalid = (cell.labels.invalid_or_unparseable ?? 0) + (cell.labels.invalid ?? 0);
  if (isMatched(condition)) {
    result.endpoint = 'per_assigned';
    result.numerator = cell.alternative;
    result.denominator = cell.assigned;
    result.rate = cell.estimate;
    result.ci = cell.interval95.alternative;
    [result.lower,result.upper] = cell.identification_bounds;
    return result;
  }
  result.endpoint = 'per_assessed';
  result.denominator = result.assessed;
  result.rate = result.assessed ? result.numerator/result.assessed : null;
  const eligible = result.assigned-result.no_evidence;
  result.lower = eligible ? result.numerator/eligible : null;
  result.upper = eligible ? (result.numerator+result.unknown)/eligible : null;
  return result;
}

function table(cs, models, capture=false) {
  if(cs.length && cs.every(isMatched))return matchedTable(cs,models,capture);
  let rows = '';
  for (const c of cs) for (const model of models) {
    const s = counts(c, model);
    if (!s) continue;
    if(capture) exportRows.push({...s,condition:c.id,model:DATA.models[model]});
    rows += `<tr><td><button data-condition="${esc(c.id)}">${esc(c.label)}</button><br><small>${esc(c.id)}</small></td><td>${esc(DATA.models[model])}</td><td><strong>${percent(s.rate)}</strong></td><td>${s.ci ? s.ci.map(percent).join("–") : "Unavailable (<2 assessed tasks)"}</td><td>${s.numerator} / ${s.assessed}</td><td>${s.mixed} / ${s.assigned}</td><td>${s.no_evidence} / ${s.assigned}</td><td>${s.unknown} / ${s.assigned}</td><td>${percent(s.lower)}–${percent(s.upper)}</td><td>${s.tasks}</td></tr>`;
  }
  return `<div class="table-scroll"><table><thead><tr><th>Condition</th><th>Model</th><th>Rate</th><th>95% CI</th><th>Against / assessed</th><th>Mixed / assigned</th><th>No evidence / assigned</th><th>Unresolved / assigned</th><th>Missingness bounds*</th><th>Source tasks</th></tr></thead><tbody>${rows}</tbody></table></div><p class="meta">* Bounds assign every unresolved outcome to following vs. non-following. Known no-evidence cases are excluded from this endpoint's denominator; all other assignments remain. No evidence is a resolved, unassessed outcome, not following or a violation. These are missingness bounds, not confidence intervals.</p>`;
}

function matchedTable(cs, models, capture) {
  let rows='';
  for(const c of cs)for(const model of models) {
    const s=counts(c,model);if(!s)continue;
    if(capture)exportRows.push({...s,condition:c.id,model:DATA.models[model]});
    rows+=`<tr><td><button data-condition="${esc(c.id)}">${esc(c.label)}</button><br><small>${esc(c.id)}</small></td><td>${esc(DATA.models[model])}</td><td>${s.numerator} / ${s.assigned}</td><td>${s.rate===null?'Withheld: unresolved':percent(s.rate)}</td><td>${s.ci?s.ci.map(percent).join('–'):'Withheld / unavailable'}</td><td class="bound-value">${percent(s.lower)} to ${percent(s.upper)}</td><td>${s.mixed} / ${s.assigned}</td><td>${s.neither} / ${s.assigned}</td><td>${s.no_evidence} / ${s.assigned}</td><td>${s.invalid} / ${s.assigned}</td><td>${s.no_implementation} / ${s.assigned}</td><td>${s.unknown} / ${s.assigned}</td><td>${s.tasks}</td></tr>`;
  }
  return `<div class="table-scroll"><table><thead><tr><th>Condition</th><th>Model</th><th>Known opposite / assigned</th><th>Rate estimate</th><th>95% task CI</th><th>Identification bounds · not a CI</th><th>Mixed / assigned</th><th>Neither / assigned</th><th>No evidence / assigned</th><th>Invalid / assigned</th><th>No implementation / assigned</th><th>Unresolved / assigned</th><th>Source tasks</th></tr></thead><tbody>${rows}</tbody></table></div><p class="meta">The denominator includes every assignment. Mixed, neither, no-evidence, invalid/unparseable and the explicitly approved no-implementation outcome are observed zeros for the narrow pure-opposite-style endpoint; this does not establish instruction compliance. Bounds assign every unknown outcome to zero versus one. They describe missing-outcome uncertainty only, not sampling uncertainty. Rate estimates and 95% task intervals are withheld where outcomes remain unresolved.</p>`;
}

function numberLine(c, models, baseline) {
  const W=700, H=Math.max(70,models.length*14+38), left=16, right=16;
  const x=v=>((left+v*(W-left-right))/W*100)+'%', axis=H-23;
  const matched=isMatched(c);
  let svg=`<svg width="100%" height="${H}" role="img" aria-label="${esc(c.label)}: ${matched?'pure opposite-style rates per assigned output; dashed diamonds are identification bounds, not confidence intervals':'rates and 95% confidence intervals'} on a 0 to 1 scale">`;
  svg+=`<line x1="${x(0)}" x2="${x(1)}" y1="${axis}" y2="${axis}" stroke="#a5afa5" ${baseline?'stroke-dasharray="6 4"':''}/>`;
  for(const value of [0,.25,.5,.75,1]) {
    svg+=`<line x1="${x(value)}" x2="${x(value)}" y1="8" y2="${axis+4}" stroke="#e5e9e1"/><text x="${x(value)}" y="${H-5}" text-anchor="middle" class="axis-label">${value}</text>`;
  }
  let visible=0;
  for(const [i,m] of models.entries()) {
    const s=counts(c,m);
    if(!s || (s.rate===null && !matched))continue;
    visible++;
    const y=14+i*14, color=modelColors[m];
    if(matched && s.unknown>0) {
      const title=`${DATA.models[m]}: ${s.numerator}/${s.assigned} known opposite-style outcomes\nIdentification bounds: ${percent(s.lower)} to ${percent(s.upper)}; NOT a 95% confidence interval\nRate estimate and CI withheld: ${s.unknown} unresolved\n${s.mixed} mixed, ${s.neither} neither, ${s.invalid} invalid / ${s.assigned} assigned`;
      svg+=`<g class="model-estimate model-bounds" tabindex="0" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title><line class="identification-bound" x1="${x(s.lower)}" x2="${x(s.upper)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="3" stroke-dasharray="3 3"/>`;
      for(const bound of [s.lower,s.upper])svg+=`<svg x="${x(bound)}" y="${y-5}" width="10" height="10" overflow="visible"><path class="bound-marker" d="M 0 0 L 4 5 L 0 10 L -4 5 Z" fill="white" stroke="${color}" stroke-width="1.8"/></svg>`;
      svg+='</g>';continue;
    }
    if(s.rate===null)continue;
    const title=`${DATA.models[m]}: ${percent(s.rate)} (${s.numerator}/${s.assessed} assessed)\n95% task-bootstrap CI: ${s.ci?s.ci.map(percent).join('–'):'unavailable'}\n${s.mixed} mixed, ${s.no_evidence} no evidence, ${s.unknown} unresolved / ${s.assigned} assigned`;
    const matchedTitle=`${DATA.models[m]}: ${percent(s.rate)} (${s.numerator}/${s.assigned} assigned)\n95% task-bootstrap CI: ${s.ci?s.ci.map(percent).join('–'):'unavailable'}\n${s.mixed} mixed, ${s.neither} neither, ${s.invalid} invalid, ${s.unknown} unresolved`;
    svg+=`<g class="model-estimate" tabindex="0" role="img" aria-label="${esc(matched?matchedTitle:title)}"><title>${esc(matched?matchedTitle:title)}</title>`;
    if(s.ci) {
      svg+=`<line class="confidence-interval" x1="${x(s.ci[0])}" x2="${x(s.ci[1])}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2.5"/>`;
      for(const bound of s.ci)svg+=`<line x1="${x(bound)}" x2="${x(bound)}" y1="${y-4}" y2="${y+4}" stroke="${color}" stroke-width="1.5"/>`;
    }
    svg+=`<circle cx="${x(s.rate)}" cy="${y}" r="4" fill="${color}" stroke="white" stroke-width="1"/></g>`;
  }
  if(!visible)svg+=`<text x="50%" y="25" text-anchor="middle" class="axis-label">No assessed outcomes</text>`;
  return svg+'</svg>';
}

function chart(group, cs, models) {
  const baselines=new Set(group.baselines||[]);
  const legend=`<div class="model-legend" aria-label="Model colors">${models.map(m=>`<span><i style="background:${modelColors[m]}"></i>${esc(DATA.models[m])}</span>`).join('')}</div>`;
  const rows=cs.map(c=>{
    const baseline=baselines.has(c.id);
    return `<div class="condition-row ${baseline?'baseline':''}"><div class="condition-label"><button data-condition="${esc(c.id)}">${esc(c.label)}${c.relation==='agree'?' · user = '+esc(isMatched(c)?c.role:'reviewer'):''}</button>${baseline?'<span class="baseline-tag">matching baseline</span>':''}<small>${esc(c.id)}</small></div><div class="number-line">${numberLine(c,models,baseline)}</div></div>`;
  }).join('');
  if(isMatched(group))return legend+'<p class="axis-caption matched-caption"><strong>Pure opposite style / all assigned outputs.</strong> Hollow diamonds joined by dashes = unknown-outcome identification bounds, <strong>not 95% confidence intervals</strong>. Filled points and solid 95% task intervals appear only for fully resolved cells. The shared absence baseline is repeated for comparison, never counted again.</p><div class="forest">'+rows+'</div>';
  return legend+'<p class="axis-caption">Rate against user preference · points = model estimates · bars = 95% confidence intervals</p><div class="forest">'+rows+'</div>';
}

function pairedFamilies(syntax=false) {
  const matched=syntax?(recoverySelected() && DATA.matched_syntax_recovery ? DATA.matched_syntax_recovery : DATA.matched_syntax):($('matched-view').value === 'prior' && DATA.matched_prior_recovery ? DATA.matched_prior_recovery : (recoverySelected()?DATA.matched_recovery:DATA.matched_review));if(!matched)return '';
  const analysis=matched.analysis, user=$('user').value;
  if(!(syntax?['all','snake_case','camelCase','single_quotes','double_quotes']:['all','loops','comprehensions']).includes(user))return '';
  const taskCount=syntax?24:32, familySizes=syntax?'6-, 12- and 6-':'12-, 15- and 3-';
  const viewNote=matched.note || (syntax?'Naming and quote robustness: each style uses its own 24 audited tasks. GLM naming remains held; the full families include both styles and all models.':recoverySelected()?'Designated recovery view: 130 fresh generations, 17 fresh blinded judgments and 86 separate free repairs; no additional observations. The original strict view remains available in the view menu.':'Original strict view: original outcomes and unresolved labels preserved.');
  const direction=user==='all'?'Both user directions, equally weighted':`User prefers ${user}`;
  let html=`<section class="matched-method"><h3>Paired task comparisons</h3><p><strong>${esc(viewNote)}</strong></p><p>Positive differences mean more pure opposite-user style in the treatment condition. For each task, subtract the reference count out of eight from the treatment count out of eight; average the two user directions, then weight all ${taskCount} tasks equally. The same absence observations are reused as a reference, not as independent new data.</p><p><strong>Bounds are not confidence intervals.</strong> They cover all assignments of unresolved labels. The frozen ${familySizes}comparison Holm families remain intact and are withheld if required outcomes are unresolved. Agreement panels are descriptive; these paired comparisons use conflicting preferences. No equivalence or motive claim follows from a small or inconclusive difference.</p><p class="meta">${esc(direction)} · 10,000 paired source-task bootstrap draws · marginal intervals, not simultaneous · Holm control is separate within each family. ${user==='all'?'':'Direction-only views are descriptive; the predeclared p-values concern the two-direction average.'}<br>Frozen analysis: ${esc(matched.analysis_source)}</p></section>`;
  for(const family of analysis.families) {
    let rows='';
    for(const comparison of family.comparisons) {
      const view=user==='all'?comparison:comparison.by_user_preference.find(v=>v.user_preference===user);
      if(!view)continue;
      const p=user==='all'?comparison.p_value:null, holm=user==='all'?comparison.holm_p_value:null;
      const inference=view.inference_status ?? comparison.inference_status;
      pairedExportRows.push({source:matched.analysis_source,study:comparison.study ?? "matched_review",family:family.name,family_size:family.holm_family_size,family_status:family.inference_status,comparison,view,user,p,holm,inference});
      rows+=`<tr><td>${esc(DATA.models[comparison.model]||comparison.model)}${comparison.study?" · "+esc(comparison.study):""}</td><td>${esc(armName(comparison.treatment_arm))}<br><span class="meta">minus ${esc(armName(comparison.reference_arm))}</span></td><td>${signedPercent(view.estimate)}</td><td>${signedRange(view.interval95)}</td><td class="bound-value">${signedRange(view.identification_bounds)}</td><td>${comparison.unresolved_required_cells}</td><td>${holm===null?'—':esc(holm.toPrecision(4))}</td><td class="inference-status">${esc(inference.replaceAll('_',' '))}${user==='all'?'':' · descriptive direction only'}</td></tr>`;
    }
    html+=`<section class="panel paired-family"><h3>${esc(familyNames[family.name]||family.name)}</h3><p class="note">${family.holm_family_size} predeclared comparisons · <strong>${esc(family.inference_status.replaceAll('_',' '))}</strong> · treatment minus reference, in percentage points.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Contrast</th><th>Difference</th><th>95% task CI</th><th>Identification bounds · not a CI</th><th>Unresolved required outcomes*</th><th>Holm p</th><th>Inference status</th></tr></thead><tbody>${rows}</tbody></table></div><p class="meta">* Count for the full two-direction comparison. A dash preserves a missing or withheld value in the frozen analysis; it is not zero, non-significance, or evidence of equivalence.</p></section>`;
  }
  return html;
}

function render() {
  const family=$('study').value, search=$('search').value.toLowerCase().trim();
  let html='',lastFamily='',count=0,matchedShown=false,syntaxShown=false;exportRows=[];pairedExportRows=[];
  $('metric').disabled=['MATCHED','MATCHED_NAMING','MATCHED_QUOTES'].includes(family);
  for(const group of DATA.groups) {
    if(family==='main' && group.family.includes('PILOT'))continue;
    if(!['all','main',group.family].includes(family))continue;
    const all=group.conditions.map(id=>conditions.get(id));
    if(search && ![group.title,DATA.families[group.family],...all.map(c=>c.id+' '+c.label)].join(' ').toLowerCase().includes(search))continue;
    const cs=all.filter(c=>modelOrder.some(m=>counts(c,m)));
    const models=modelOrder.filter(m=>cs.some(c=>counts(c,m)));
    if(!models.length)continue;
    if(group.family!==lastFamily){html+=`<h2 class="family-title">${esc(DATA.families[group.family])}</h2>`;lastFamily=group.family;}
    if(group.family==='MATCHED' && !matchedShown){html+=pairedFamilies();matchedShown=true;}
    if(['MATCHED_NAMING','MATCHED_QUOTES'].includes(group.family) && !syntaxShown){html+=pairedFamilies(true);syntaxShown=true;}
    const unknown=cs.reduce((n,c)=>n+models.reduce((a,m)=>a+(counts(c,m)?.unknown||0),0),0);
    const noEvidence=cs.reduce((n,c)=>n+models.reduce((a,m)=>a+(counts(c,m)?.no_evidence||0),0),0);
    html+=`<section class="panel" id="${group.id}"><h3>${esc(group.title)}</h3><p class="note">${esc(group.note)}</p>${unknown?`<span class="badge">${fmt(unknown)} unresolved outcomes in this panel</span>`:''}${noEvidence?`<span class="badge">${fmt(noEvidence)} resolved no-evidence outcomes in this panel</span>`:''}${group.missing_baselines?.length?`<p class="warning">No matching baseline: ${group.missing_baselines.map(id=>esc(conditions.get(id).label)).join('; ')}.</p>`:''}${chart(group,cs,models)}<details><summary>Exact rates, 95% intervals &amp; missingness</summary>${table(cs,models,true)}</details></section>`;
    count++;
  }
  $('charts').innerHTML=html;$('empty').hidden=count>0;
  $('export-paired').hidden=!pairedExportRows.length;
}

function showPrompt(id, updateHash=true) {
  const c=conditions.get(id);if(!c)return;
  currentCondition=c;currentPrompt=0;templateMode=true;
  $('prompt-title').textContent=c.label;$('prompt-id').textContent=c.id;
  const actor=isMatched(c) && c.role && c.role!=='absent'?c.role:'reviewer';
  $('prompt-description').textContent=`${DATA.families[c.family]} · Mode ${c.mode} · ${c.scenario.replaceAll('_',' ')} · ${actor} ${c.relation==='absent'?'not mentioned':c.relation==='agree'?'agrees with user':'opposes user'}. Charts use the selected user-direction view; select an exact saved combination below.`;
  $('prompt-example').innerHTML=c.prompts.map((p,i)=>`<option value="${i}">${esc('User: '+p.condition.user_preference+' · '+actor+': '+(p.condition.grader_preference||'none')+(p.condition.reward_preference?' · reward: '+p.condition.reward_preference:'')+' — '+p.source.split('/').slice(1,3).join('/'))}</option>`).join('');
  $('prompt-counts').innerHTML=table([c],modelOrder);
  updatePrompt();
  if(!$('prompt-dialog').open)$('prompt-dialog').showModal();
  if(updateHash)history.replaceState(null,'','#condition='+encodeURIComponent(id));
}
function updatePrompt() {
  const p=currentCondition.prompts[currentPrompt];
  $('prompt-text').textContent=p[templateMode?'template':'example'];
  $('prompt-meta').textContent=`${templateMode?'Template view: only task source is replaced with {{TASK_SOURCE}}. Preference wording is preserved.':'Exact saved prompt example.'}\nTask: ${p.task}\nSource: ${p.source}\nAssignment: ${p.assignment}${p.sha256?'\nSaved prompt SHA-256: '+p.sha256:''}`;
  $('template-tab').setAttribute('aria-pressed',String(templateMode));
  $('example-tab').setAttribute('aria-pressed',String(!templateMode));
  $('copy-prompt').textContent='Copy text';
}
$('study').insertAdjacentHTML('beforeend',Object.entries(DATA.families).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join(''));
$('snapshot').textContent='SNAPSHOT · '+new Date(DATA.generated).toLocaleString();
$('summary').innerHTML=`<span><strong>${fmt(DATA.assigned)}</strong>assigned observations</span><span><strong>${DATA.conditions.length}</strong>condition families</span><span><strong>${Object.keys(DATA.models).length}</strong>models across studies</span><span><strong>${DATA.groups.length}</strong>comparison panels</span>`;
$('sources').innerHTML='<details><summary>Source files &amp; snapshot coverage</summary>'+DATA.sources.map(s=>`<p class="mono">${esc(s.path)} · ${fmt(s.rows)} rows<br>${esc(s.judge)}<br>SHA-256 ${s.sha256}</p>`).join('')+(DATA.missing.length?'<p class="warning">Unavailable: '+DATA.missing.map(esc).join(', ')+'</p>':'')+'</details>';
for(const id of ['study','user','metric'])$(id).addEventListener('change',render);
$('search').addEventListener('input',render);
document.addEventListener('click',e=>{const button=e.target.closest('[data-condition]');if(button)showPrompt(button.dataset.condition);});
$('prompt-example').addEventListener('change',e=>{currentPrompt=Number(e.target.value);updatePrompt();});
$('template-tab').addEventListener('click',()=>{templateMode=true;updatePrompt();});
$('example-tab').addEventListener('click',()=>{templateMode=false;updatePrompt();});
$('close-dialog').addEventListener('click',()=>$('prompt-dialog').close());
$('prompt-dialog').addEventListener('close',()=>history.replaceState(null,'',location.pathname+location.search));
$('copy-prompt').addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText($('prompt-text').textContent);$('copy-prompt').textContent='Copied';}
  catch{const range=document.createRange();range.selectNodeContents($('prompt-text'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);$('copy-prompt').textContent='Selected · press ⌘C / Ctrl+C';}
});
function rateCSV() {
  const unique=[...new Map(exportRows.map(r=>[r.condition+'|'+r.model,r])).values()];
  const header=['condition','model','metric','user_preference','rate','numerator','assessed','resolved','mixed','no_evidence','unknown','assigned','ci95_low','ci95_high','missingness_lower','missingness_upper','sources','endpoint','denominator','neither','invalid','no_implementation'];
  const rows=unique.map(r=>[r.condition,r.model,r.endpoint==='per_assigned'?'opposite_pure_style':$('metric').value,$('user').value,r.rate??'',r.numerator,r.assessed,r.resolved,r.mixed,r.no_evidence,r.unknown,r.assigned,r.ci?.[0]??'',r.ci?.[1]??'',r.lower??'',r.upper??'',r.sources.join('; '),r.endpoint,r.denominator,r.neither,r.invalid,r.no_implementation]);
  return csvText([header,...rows]);
}
function pairedCSV() {
  const header=['family','family_size','family_status','model','contrast','treatment_arm','reference_arm','reviewer_relation','user_preference','estimate','ci95_low','ci95_high','identification_lower','identification_upper','p_value','holm_p_value','reject_at_05','inference_status','unresolved_required_outcomes_both_directions','source','units','view_kind','degenerate_task_variance','study'];
  const rows=pairedExportRows.map(r=>[r.family,r.family_size,r.family_status,r.comparison.model,r.comparison.contrast,r.comparison.treatment_arm,r.comparison.reference_arm,r.comparison.reviewer_relation??"conflict_where_present",r.user,r.view.estimate??'',r.view.interval95?.[0]??'',r.view.interval95?.[1]??'',r.view.identification_bounds?.[0]??'',r.view.identification_bounds?.[1]??'',r.p??'',r.holm??'',r.user==='all'?(r.comparison.reject_at_05??''):'',r.inference,r.comparison.unresolved_required_cells,r.source,'proportion_difference',r.user==='all'?'predeclared_two_direction':'descriptive_direction',r.view.degenerate_task_variance??'',r.study]);
  return csvText([header,...rows]);
}
function csvText(rows) {
  return rows.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
}
function downloadCSV(csv,filename) {
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('export').addEventListener('click',()=>{
  downloadCSV(rateCSV(),'grader-power-rates.csv');
});
$('export-paired').addEventListener('click',()=>{
  downloadCSV(pairedCSV(),'grader-power-matched-paired-comparisons.csv');
});
$('matched-view').disabled=!DATA.matched_recovery;
$('matched-view').addEventListener('change',render);
render();
if(location.hash.startsWith('#condition='))showPrompt(decodeURIComponent(location.hash.slice(11)),false);
