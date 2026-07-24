/**
 * Interactive data cleaning strategies widget
 * Shows 9 strategies with survival rates and statistics
 */

export function renderCleaningWidget(containerId, strategies, finalStats) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Build HTML
  let html = `
    <div class="cleaning-strategies">
      <h3>🧹 Data Cleaning Pipeline</h3>
      <p class="widget-subtitle">Watch 22.2M rows get filtered through 9 production strategies</p>
      
      <div class="pipeline-viz">
        ${strategies.map((s, idx) => `
          <div class="strategy-step" data-step="${idx}">
            <div class="step-header">
              <div class="step-number">${s.id}</div>
              <div class="step-info">
                <div class="step-name">${s.name}</div>
                <div class="step-desc">${s.description}</div>
              </div>
              <div class="step-survival">${s.survival.toFixed(2)}%</div>
            </div>
            
            <div class="step-details">
              <div class="detail-row">
                <span class="detail-label">Applies to:</span>
                <span class="detail-value">${s.appliesTo}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Removed:</span>
                <span class="detail-value removed">${formatNumber(s.removed)} rows</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Reason:</span>
                <span class="detail-value">${s.reason}</span>
              </div>
            </div>
            
            <div class="survival-bar">
              <div class="survival-fill" style="width: ${s.survival}%"></div>
            </div>
          </div>
        `).join('')}
      </div>
      
      <div class="final-stats-box">
        <h4>📊 Final Statistics</h4>
        <div class="stats-row">
          <div class="stat-item">
            <div class="stat-icon">📄</div>
            <div class="stat-content">
              <div class="stat-number">${formatLargeNumber(finalStats.originalRows)} → ${formatLargeNumber(finalStats.finalRows)}</div>
              <div class="stat-label">Rows (${finalStats.survivalRate}% survival)</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon">💬</div>
            <div class="stat-content">
              <div class="stat-number">${formatLargeNumber(finalStats.originalTokens)} → ${formatLargeNumber(finalStats.finalTokens)}</div>
              <div class="stat-label">Tokens (${((finalStats.finalTokens / finalStats.originalTokens) * 100).toFixed(2)}% survival)</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon">🗑️</div>
            <div class="stat-content">
              <div class="stat-number">${formatLargeNumber(finalStats.originalRows - finalStats.finalRows)}</div>
              <div class="stat-label">Removed (${(100 - finalStats.survivalRate).toFixed(2)}%)</div>
            </div>
          </div>
          <div class="stat-item">
            <div class="stat-icon">⏱️</div>
            <div class="stat-content">
              <div class="stat-number">${finalStats.cleaningTimeHours}h</div>
              <div class="stat-label">Processing time</div>
            </div>
          </div>
        </div>
        <div class="hardware-info">
          <strong>Hardware:</strong> ${finalStats.hardwareUsed}
        </div>
      </div>
      
      <div class="interaction-hint">
        <span class="hint-icon">💡</span>
        <span class="hint-text">Click any strategy to see detailed statistics</span>
      </div>
    </div>
  `;

  container.innerHTML = html;

  // Add interactivity
  const steps = container.querySelectorAll('.strategy-step');
  steps.forEach((step, idx) => {
    step.addEventListener('click', () => {
      // Toggle active state
      const isActive = step.classList.contains('active');
      steps.forEach(s => s.classList.remove('active'));
      if (!isActive) {
        step.classList.add('active');
      }
    });

    // Animate on scroll
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setTimeout(() => {
            entry.target.classList.add('visible');
          }, idx * 100);
        }
      });
    }, { threshold: 0.1 });

    observer.observe(step);
  });
}

function formatNumber(num) {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(2) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + 'K';
  }
  return num.toString();
}

function formatLargeNumber(num) {
  if (num >= 1000000000) {
    return (num / 1000000000).toFixed(1) + 'B';
  } else if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + 'K';
  }
  return num.toString();
}
