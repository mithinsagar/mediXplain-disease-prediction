/* -------------------------------------------------------------------------
   MediXplain front-end script
   Author:  Mithin Sagar S
   GitHub:  https://github.com/mithinsagar
   Project: MediXplain - Explainable Disease Prediction System
   ------------------------------------------------------------------------- */

const MediXplain = (function () {
    const symptomItems = () => document.querySelectorAll('.symptom-item');
    const symptomCheckboxes = () => document.querySelectorAll('input[name="symptoms"]');

    function updateCounter() {
        const badge = document.getElementById('counterBadge');
        if (!badge) return;
        const count = document.querySelectorAll('input[name="symptoms"]:checked').length;
        badge.textContent = count + ' selected';
        badge.classList.toggle('has-items', count > 0);
    }

    function clearAll() {
        symptomCheckboxes().forEach(cb => (cb.checked = false));
        updateCounter();
    }

    function attachSearch() {
        const input = document.getElementById('symptomSearch');
        if (!input) return;

        symptomItems().forEach(el => (el._name = el.getAttribute('data-name') || ''));

        input.addEventListener('input', function () {
            const q = this.value.toLowerCase().trim();
            symptomItems().forEach(el => {
                el.style.display = !q || el._name.includes(q) ? '' : 'none';
            });
        });
    }

    function animateConfidence() {
        const fill = document.getElementById('confFill');
        if (!fill) return;
        const target = fill.dataset.target || 0;
        requestAnimationFrame(() => {
            setTimeout(() => {
                fill.style.width = target + '%';
            }, 80);
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        attachSearch();
        symptomCheckboxes().forEach(cb => cb.addEventListener('change', updateCounter));
        updateCounter();
        animateConfidence();
    });

    return { clearAll, updateCounter };
})();
