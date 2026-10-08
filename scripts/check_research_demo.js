// Лёгкая проверка логики всех вариантов демонстрации без запуска браузера.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const page = fs.readFileSync(path.join(root, 'presentation/research_demo.html'), 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
function element() {
  return { value: '0', children: [], innerHTML: '', textContent: '',
    append(e) { this.children.push(e); }, replaceChildren() { this.children = []; },
    addEventListener() {} };
}
const ids = Object.fromEntries(['case', 'chart', 'note', 'forecast'].map(id => [id, element()]));
const context = vm.createContext({ document: { getElementById: id => ids[id], createElement: element } });
vm.runInContext(script + '\nglobalThis.caseCount = cases.length; globalThis.renderCase = render;', context);
const checks = [];
for (let i = 0; i < context.caseCount; i++) {
  ids.case.value = String(i);
  context.renderCase();
  const passed = ids.forecast.children.length === 12 && ids.chart.innerHTML.includes('<path') &&
    !ids.chart.innerHTML.includes('NaN') && ids.note.textContent.includes('Детектор:');
  checks.push({ case: i + 1, passed, forecast_rows: ids.forecast.children.length });
}
const passed = checks.length > 0 && checks.every(c => c.passed);
fs.writeFileSync(path.join(root, 'tracking/research_demo_check.json'), JSON.stringify({
  passed, checks, scope: 'JavaScript logic with DOM stub; browser rendering not verified'
}, null, 2));
console.log(`Demo logic: ${checks.filter(c => c.passed).length}/${checks.length} cases passed`);
if (!passed) process.exit(1);
