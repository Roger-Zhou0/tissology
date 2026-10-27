'use strict';
(() => {
  const guide = document.getElementById('user-guide');
  function openGuide() {
    if (!guide.open) guide.showModal();
    guide.scrollTop = 0;
  }
  document.querySelectorAll('.guide-trigger').forEach(button => button.addEventListener('click', openGuide));
  document.getElementById('close-guide').addEventListener('click', () => guide.close());
  document.getElementById('guide-done').addEventListener('click', () => guide.close());
  guide.querySelectorAll('[data-guide-route]').forEach(button => button.addEventListener('click', () => {
    guide.close();
    location.hash = button.dataset.guideRoute;
  }));
  if (new URLSearchParams(location.search).get('guide') === '1') openGuide();
})();
