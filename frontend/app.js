document.addEventListener('DOMContentLoaded', async () => {

    // ── Experiment Metadata ───────────────────────────────────────────────────
    const EXP_META = {
        exp_rep:    { icon: '📊', desc: 'Tests all 10 input representations against selected ciphers/models/rounds. Identifies which data format gives the best distinguishing accuracy.' },
        exp_model:  { icon: '🧠', desc: 'Compares MLP, CNN, SiameseNet, and MINE architectures on the same cipher/rep/round configuration.' },
        exp_round:  { icon: '🔁', desc: 'Sweeps across multiple round counts to find the security limit — the round where accuracy drops to ~50% (random guessing).' },
        exp_cm:     { icon: '🔢', desc: 'Generates a confusion matrix heatmap showing true vs predicted labels. Select multiple ciphers to compare side-by-side.' },
        exp_dist:   { icon: '📈', desc: 'Plots the Hamming weight distribution of ΔC = C ⊕ C\' and compares it to the ideal random (binomial) distribution.' },
        bonus_diff: { icon: '🔍', desc: 'Random search for high-probability differential characteristics. Plots trial scores and reports the best ΔP found.' },
        bonus_class:{ icon: '⚖️',  desc: 'Runs classical differential cryptanalysis and estimates ML distinguisher advantage on the same cipher.' },
        bonus_trans:{ icon: '🔀', desc: 'Pre-trains on fewer rounds (source), fine-tunes on more rounds (target). Compares against training from scratch.' },
        bonus_key:  { icon: '🔑', desc: 'Trains a distinguisher then uses it to score 256 partial subkey candidates. Shows the key score distribution.' },
    };

    // ── Elements ──────────────────────────────────────────────────────────────
    let ciphers = [];
    const els = {
        expType:        document.getElementById('exp-type'),
        cipherSel:      document.getElementById('cipher-select'),
        modelSel:       document.getElementById('model-select'),
        repSel:         document.getElementById('rep-select'),
        roundsIn:       document.getElementById('rounds-input'),
        samplesIn:      document.getElementById('samples-input'),
        epochsIn:       document.getElementById('epochs-input'),
        srcRoundsIn:    document.getElementById('src-rounds-input'),
        roundSweepIn:   document.getElementById('round-sweep-input'),
        runBtn:         document.getElementById('run-btn'),
        pipelineBtn:    document.getElementById('pipeline-btn'),

        // Desc card
        expDescCard:    document.getElementById('exp-desc-card'),
        expDescIcon:    document.getElementById('exp-desc-icon'),
        expDescText:    document.getElementById('exp-desc-text'),

        // Groups
        cipherGroup:    document.getElementById('cipher-group'),
        modelGroup:     document.getElementById('model-group'),
        repGroup:       document.getElementById('rep-group'),
        roundSweepGrp:  document.getElementById('round-sweep-group'),
        singleRoundRow: document.getElementById('single-round-row'),
        epochsRow:      document.getElementById('epochs-row'),
        srcRoundsGrp:   document.getElementById('src-rounds-group'),
        samplesLabel:   document.getElementById('samples-label'),

        // Context panel
        ctxPanel:       document.getElementById('context-panel'),
        cipherInfo:     document.getElementById('cipher-info'),

        // Pipeline
        pipelinePanel:  document.getElementById('pipeline-status-panel'),
        pipeSteps:      [1,2,3,4].map(i => document.getElementById(`pipe-step-${i}`)),

        // Views
        jobView:        document.getElementById('job-view'),
        jobTitle:       document.getElementById('job-title'),
        resView:        document.getElementById('results-view'),
        pipelineResView:document.getElementById('pipeline-results-view'),
        placeholder:    document.getElementById('placeholder-msg'),
        resContent:     document.getElementById('results-content'),

        // Job
        progressFill:   document.getElementById('progress-fill'),
        progressText:   document.getElementById('progress-text'),
        logConsole:     document.getElementById('log-console'),
        jobIdLabel:     document.getElementById('current-job-id'),

        // Results
        resPlot:        document.getElementById('result-plot'),
        resStats:       document.getElementById('results-stats'),
        resTitle:       document.getElementById('results-title'),
        resSub:         document.getElementById('results-subtitle'),
        rawDataCont:    document.getElementById('raw-data-container'),
        rawData:        document.getElementById('raw-data'),

        // Table
        dataTblCont:    document.getElementById('data-table-container'),
        resultsTable:   document.getElementById('results-table'),
        tblHead:        document.getElementById('results-table-head'),
        tblBody:        document.getElementById('results-table-body'),
        copyTableBtn:   document.getElementById('copy-table-btn'),

        // Pipeline grid
        pipelineGrid:   document.getElementById('pipeline-grid'),
    };

    // ── Init ──────────────────────────────────────────────────────────────────
    async function init() {
        try {
            const [cRes, mRes, rRes] = await Promise.all([
                fetch('/api/ciphers').then(r => r.json()),
                fetch('/api/models').then(r => r.json()),
                fetch('/api/representations').then(r => r.json()),
            ]);

            ciphers = cRes;
            els.cipherSel.innerHTML = cRes.map(c =>
                `<option value="${c.name}">${c.name.toUpperCase()}</option>`).join('');
            els.modelSel.innerHTML = mRes.map(m =>
                `<option value="${m}">${m}</option>`).join('');
            els.repSel.innerHTML = rRes.map(r =>
                `<option value="${r.id}">${r.id}: ${r.label}</option>`).join('');

            // Smart defaults matching run_all.py
            selectOptions(els.cipherSel, ['craft']);
            selectOptions(els.modelSel,  ['MLP']);
            selectOptions(els.repSel,    ['2']);

            updateDescCard();
            updateContextPanel();
            updateFormState();

        } catch (e) {
            console.error('Failed to load metadata', e);
            alert('Failed to connect to backend. Is the server running?\n\n' + e.message);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    function selectOptions(sel, values) {
        for (const opt of sel.options) {
            opt.selected = values.includes(opt.value);
        }
    }

    function getSelected(sel) {
        return Array.from(sel.selectedOptions).map(o => o.value);
    }

    function getSelectedInts(sel) {
        return Array.from(sel.selectedOptions).map(o => parseInt(o.value));
    }

    function show(el, yes) { el.style.display = yes ? '' : 'none'; }

    // ── UI Updates ────────────────────────────────────────────────────────────
    els.expType.addEventListener('change', () => { updateDescCard(); updateFormState(); });
    els.cipherSel.addEventListener('change', updateContextPanel);

    function updateDescCard() {
        const meta = EXP_META[els.expType.value];
        if (!meta) return;
        els.expDescIcon.textContent = meta.icon;
        els.expDescText.textContent = meta.desc;
    }

    function updateContextPanel() {
        const selected = getSelected(els.cipherSel);
        if (selected.length !== 1) { els.ctxPanel.style.display = 'none'; return; }
        const c = ciphers.find(x => x.name === selected[0]);
        if (!c) return;
        els.ctxPanel.style.display = 'block';
        els.cipherInfo.innerHTML = `
            <span>Year:</span>      <span>${c.year || '—'}</span>
            <span>Structure:</span> <span>${c.structure || '—'}</span>
            <span>Block Size:</span><span>${c.block ? c.block + '-bit' : '—'}</span>
        `;
    }

    function updateFormState() {
        const type = els.expType.value;
        const isML    = !['exp_dist','bonus_diff'].includes(type);
        const isMulti = ['exp_rep','exp_model','exp_round','exp_cm'].includes(type);
        const isRound = type === 'exp_round';
        const isTrans = type === 'bonus_trans';
        const isDiff  = type === 'bonus_diff';

        show(els.modelGroup,     !['exp_dist','bonus_diff'].includes(type));
        show(els.repGroup,       !['exp_dist','bonus_diff','bonus_class','bonus_key'].includes(type));
        show(els.roundSweepGrp,  isRound);
        show(els.singleRoundRow, !isRound);
        show(els.epochsRow,      isML);
        show(els.srcRoundsGrp,   isTrans);

        els.samplesLabel.textContent = isDiff ? 'Trials' : 'Samples';
        if (isDiff && parseInt(els.samplesIn.value) > 100) els.samplesIn.value = 30;
        if (!isDiff && parseInt(els.samplesIn.value) < 100) els.samplesIn.value = 2000;

        // Allow multi-select for main experiments
        els.cipherSel.size = isMulti ? 5 : 3;
        els.modelSel.size  = isMulti ? 4 : 2;
    }

    // ── Run Single Experiment ─────────────────────────────────────────────────
    els.runBtn.addEventListener('click', async () => {
        const type = els.expType.value;
        const payload = buildPayload(type);
        const endpoint = getEndpoint(type);

        disableButtons();
        showJobView(`Running ${getExpTitle(type)}...`);

        try {
            const res  = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (data.job_id) {
                monitorJob(data.job_id, (result) => {
                    showSingleResult(result, type);
                    resetButtons();
                });
            } else {
                throw new Error(data.detail || 'No job ID returned');
            }
        } catch (e) {
            alert('Failed to start experiment: ' + e.message);
            resetButtons();
        }
    });

    // ── Run Full Pipeline ─────────────────────────────────────────────────────
    els.pipelineBtn.addEventListener('click', async () => {
        disableButtons();
        els.pipelinePanel.style.display = 'block';
        els.pipelineResView.style.display = 'none';
        els.pipelineGrid.innerHTML = '';
        els.pipeSteps.forEach(s => s.dataset.state = 'pending');

        // Collect params (use current selection)
        const cipherList = getSelected(els.cipherSel);
        const modelList  = getSelected(els.modelSel);
        const repList    = getSelectedInts(els.repSel);
        const rounds     = parseInt(els.roundsIn.value);
        const samples    = parseInt(els.samplesIn.value);
        const epochs     = parseInt(els.epochsIn.value);

        const steps = [
            {
                label: '① Representation Analysis',
                endpoint: '/api/exp/representation',
                payload: { ciphers: cipherList, models: modelList, rounds: [rounds], rep_ids: [1,2,3,4,5,6,7,8,9,10], n_samples: samples, epochs },
            },
            {
                label: '② Model Architecture Comparison',
                endpoint: '/api/exp/model',
                payload: { ciphers: cipherList, models: ['MLP','CNN','SiameseNet','MINE'], rounds: [rounds], rep_ids: repList, n_samples: samples, epochs },
            },
            {
                label: '③ Round Limits Analysis',
                endpoint: '/api/exp/round',
                payload: { ciphers: cipherList, models: modelList, rep_ids: repList, round_list: [3,4,5,6,7], n_samples: samples, epochs },
            },
            {
                label: '④ Confusion Matrix',
                endpoint: '/api/exp/confusion',
                payload: { ciphers: cipherList, model_type: modelList[0] || 'MLP', rep_id: repList[0] || 2, rounds, n_samples: samples, epochs },
            },
        ];

        const allResults = [];

        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            els.pipeSteps[i].dataset.state = 'running';
            showJobView(step.label);

            try {
                const res  = await fetch(step.endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(step.payload),
                });
                const data = await res.json();
                if (!data.job_id) throw new Error(data.detail || 'No job ID');

                const result = await monitorJobAsync(data.job_id);
                els.pipeSteps[i].dataset.state = 'done';
                allResults.push({ label: step.label, result });

                // Add card to pipeline grid
                const card = document.createElement('div');
                card.className = 'pipeline-result-card';
                card.innerHTML = `
                    <div class="card-header">${step.label}</div>
                    <img src="${result.plot}" alt="${step.label}">
                `;
                els.pipelineGrid.appendChild(card);

            } catch (e) {
                els.pipeSteps[i].dataset.state = 'error';
                appendLog(`❌ ${step.label} failed: ${e.message}`);
                // Continue to next step even on failure
            }
        }

        // Show pipeline results
        els.jobView.style.display = 'none';
        els.resView.style.display = 'none';
        els.pipelineResView.style.display = 'flex';
        resetButtons();
    });

    // ── Payload Builder ───────────────────────────────────────────────────────
    function buildPayload(type) {
        const c   = getSelected(els.cipherSel);
        const m   = getSelected(els.modelSel);
        const r   = getSelectedInts(els.repSel);
        const rnd = parseInt(els.roundsIn.value);
        const samp = parseInt(els.samplesIn.value);
        const ep   = parseInt(els.epochsIn.value);

        switch (type) {
            case 'exp_rep':
                return { ciphers: c, models: m, rounds: [rnd], rep_ids: r.length ? r : [1,2,3,4,5,6,7,8,9,10], n_samples: samp, epochs: ep };
            case 'exp_model':
                return { ciphers: c, models: ['MLP','CNN','SiameseNet','MINE'], rounds: [rnd], rep_ids: r, n_samples: samp, epochs: ep };
            case 'exp_round': {
                const sweepStr = els.roundSweepIn.value;
                const sweepList = sweepStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                return { ciphers: c, models: m, rep_ids: r, round_list: sweepList.length ? sweepList : [3,4,5,6,7], n_samples: samp, epochs: ep };
            }
            case 'exp_cm':
                return { ciphers: c, model_type: m[0] || 'MLP', rep_id: r[0] || 2, rounds: rnd, n_samples: samp, epochs: ep };
            case 'exp_dist':
                return { cipher_name: c[0] || 'craft', rounds: rnd, n_samples: samp };
            case 'bonus_diff':
                return { cipher_name: c[0] || 'craft', rounds: rnd, num_trials: samp };
            case 'bonus_class':
                return { cipher_name: c[0] || 'craft', rounds: rnd, n_samples: samp };
            case 'bonus_trans':
                return { cipher_name: c[0] || 'craft', source_rounds: parseInt(els.srcRoundsIn.value), target_rounds: rnd, rep_id: r[0] || 2, n_samples: samp, epochs: ep };
            case 'bonus_key':
                return { cipher_name: c[0] || 'craft', rounds: rnd, n_samples: samp, epochs: ep };
            default:
                return {};
        }
    }

    function getEndpoint(type) {
        return {
            exp_rep:    '/api/exp/representation',
            exp_model:  '/api/exp/model',
            exp_round:  '/api/exp/round',
            exp_cm:     '/api/exp/confusion',
            exp_dist:   '/api/exp/distribution',
            bonus_diff: '/api/bonus/diff-search',
            bonus_class:'/api/bonus/classical',
            bonus_trans:'/api/bonus/transfer',
            bonus_key:  '/api/bonus/key-recovery',
        }[type];
    }

    function getExpTitle(type) {
        return {
            exp_rep:    'Representation Analysis',
            exp_model:  'Model Comparison',
            exp_round:  'Round Analysis',
            exp_cm:     'Confusion Matrix',
            exp_dist:   'Dataset Distribution',
            bonus_diff: 'Difference Search',
            bonus_class:'Classical vs ML',
            bonus_trans:'Transfer Learning',
            bonus_key:  'Key Recovery',
        }[type] || 'Experiment';
    }

    // ── Job Monitoring ────────────────────────────────────────────────────────
    function monitorJob(jobId, onDone) {
        els.jobIdLabel.textContent = `Job: ${jobId}`;
        const evtSource = new EventSource(`/api/progress/${jobId}`);

        evtSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.type === 'ping') return;

            if (data.message) appendLog(data.message);
            if (data.progress !== undefined) setProgress(data.progress);

            if (data.type === 'done') {
                evtSource.close();
                fetchAndReturn(jobId, onDone);
            } else if (data.type === 'error') {
                evtSource.close();
                resetButtons();
            }
        };
        evtSource.onerror = () => { evtSource.close(); resetButtons(); };
    }

    // Promise-based version for pipeline
    function monitorJobAsync(jobId) {
        return new Promise((resolve, reject) => {
            els.jobIdLabel.textContent = `Job: ${jobId}`;
            const evtSource = new EventSource(`/api/progress/${jobId}`);

            evtSource.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'ping') return;
                if (data.message) appendLog(data.message);
                if (data.progress !== undefined) setProgress(data.progress);

                if (data.type === 'done') {
                    evtSource.close();
                    fetch(`/api/results/${jobId}`)
                        .then(r => r.json())
                        .then(d => {
                            if (d.status === 'done') resolve(d.result);
                            else reject(new Error('Job did not complete: ' + d.status));
                        })
                        .catch(reject);
                } else if (data.type === 'error') {
                    evtSource.close();
                    reject(new Error(data.message));
                }
            };
            evtSource.onerror = () => { evtSource.close(); reject(new Error('SSE error')); };
        });
    }

    async function fetchAndReturn(jobId, onDone) {
        try {
            const res  = await fetch(`/api/results/${jobId}`);
            const data = await res.json();
            if (data.status === 'done') onDone(data.result);
        } catch (e) {
            console.error('fetchResults error', e);
        }
    }

    // ── View Helpers ──────────────────────────────────────────────────────────
    function showJobView(title) {
        els.jobTitle.textContent = title;
        els.jobView.style.display = 'flex';
        els.resView.style.display = 'none';
        els.pipelineResView.style.display = 'none';
        els.logConsole.innerHTML = '';
        setProgress(0);
    }

    function appendLog(msg) {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const div  = document.createElement('div');
        div.className = 'log-entry';
        let html = msg
            .replace(/✅/g, '<span style="color:var(--accent-green)">✅</span>')
            .replace(/❌/g, '<span style="color:var(--accent-red)">❌</span>')
            .replace(/🔑|📊|🔧|🧠|⚙️|🔁|🔬|🔍|⚖️|🔢/g, '<span>$&</span>');
        div.innerHTML = `<span class="log-time">[${time}]</span> ${html}`;
        els.logConsole.appendChild(div);
        els.logConsole.scrollTop = els.logConsole.scrollHeight;
    }

    function setProgress(pct) {
        els.progressFill.style.width = `${pct}%`;
        els.progressText.textContent = `${pct}%`;
    }

    // ── Render Single Result ──────────────────────────────────────────────────
    function showSingleResult(result, type) {
        els.jobView.style.display = 'none';
        els.resView.style.display = 'flex';
        els.placeholder.style.display = 'none';
        els.resContent.style.display = 'block';

        els.resTitle.textContent   = getExpTitle(type);
        els.resSub.textContent     = new Date().toLocaleString();
        els.resPlot.src            = result.plot;

        // Stat pills
        let statsHtml = '';
        if (result.improvement !== undefined) {
            const pct = (result.improvement * 100).toFixed(1);
            statsHtml = `<span class="stat-pill ${pct >= 0 ? '' : 'red'}">${pct >= 0 ? '+' : ''}${pct}% via Transfer</span>`;
        } else if (result.match !== undefined) {
            statsHtml = `<span class="stat-pill ${result.match ? '' : 'red'}">Key Match: ${result.match ? '✅ Success' : '❌ Failed'}</span>`;
        } else if (result.best_score !== undefined) {
            statsHtml = `<span class="stat-pill">Best Score: ${result.best_score}</span>`;
        } else if (result.advantage !== undefined) {
            statsHtml = `<span class="stat-pill">Advantage: ${result.advantage}</span>`;
        }
        els.resStats.innerHTML = statsHtml;

        // Data table
        const tableData = result.data || result.trials;
        if (tableData && Array.isArray(tableData) && tableData.length > 0) {
            renderTable(tableData);
            show(els.dataTblCont, true);
            show(els.rawDataCont, false);
        } else {
            show(els.dataTblCont, false);
            // Show remaining result fields as JSON
            const displayResult = { ...result };
            delete displayResult.plot;
            delete displayResult.data;
            delete displayResult.trials;
            delete displayResult.scores;
            if (Object.keys(displayResult).length > 0) {
                els.rawData.textContent = JSON.stringify(displayResult, null, 2);
                show(els.rawDataCont, true);
            } else {
                show(els.rawDataCont, false);
            }
        }
    }

    // ── Results Table Renderer ────────────────────────────────────────────────
    let _tableCsv = '';

    function renderTable(rows) {
        if (!rows || rows.length === 0) return;
        const keys = Object.keys(rows[0]);
        const accKey = keys.find(k => k.toLowerCase().includes('acc') || k.toLowerCase().includes('accuracy'));

        // Header
        els.tblHead.innerHTML = `<tr>${keys.map(k => `<th>${k}</th>`).join('')}</tr>`;

        // Body
        els.tblBody.innerHTML = rows.map(row => {
            return `<tr>${keys.map(k => {
                const v = row[k];
                let cls = '';
                if (k === accKey && typeof v === 'number') {
                    cls = v >= 0.7 ? 'acc-cell acc-high' : v >= 0.55 ? 'acc-cell acc-mid' : 'acc-cell acc-low';
                    return `<td class="${cls}">${v.toFixed(4)}</td>`;
                }
                return `<td>${v}</td>`;
            }).join('')}</tr>`;
        }).join('');

        // Build CSV for copy
        _tableCsv = [keys.join(','), ...rows.map(r => keys.map(k => r[k]).join(','))].join('\n');
    }

    els.copyTableBtn.addEventListener('click', () => {
        if (!_tableCsv) return;
        const blob = new Blob([_tableCsv], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `results_${els.expType.value}_${Date.now()}.csv`;
        a.click();
    });

    // ── Button State ──────────────────────────────────────────────────────────
    function disableButtons() {
        els.runBtn.disabled = true;
        els.pipelineBtn.disabled = true;
        els.runBtn.innerHTML = '<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span>&nbsp;Running...';
    }

    function resetButtons() {
        els.runBtn.disabled = false;
        els.pipelineBtn.disabled = false;
        els.runBtn.innerHTML = '<span class="btn-icon">▶</span> Run Experiment';
    }

    // ── Start ─────────────────────────────────────────────────────────────────
    init();
});
