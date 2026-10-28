'use strict';
(() => {
  const entries = [
    ['reference-label', 'Reference label', 'The category assigned to an image in the dataset. It is the comparison point for evaluating a model; it is separate from the model’s prediction and your own review.'],
    ['prediction', 'Prediction', 'The category a model suggests for an image. A prediction can disagree with the reference label. Your review records your judgment without changing either one.'],
    ['score', 'Model score', 'A number used by the model to rank its possible labels. The local classifier’s scores add up to 100%, but a score of 90% does not mean the prediction has a verified 90% chance of being correct. These scores are not calibrated clinical confidence.'],
    ['accuracy', 'Label agreement (accuracy)', 'Out of all evaluated images, the fraction whose predicted label matches the reference label. For example, 90 matches out of 100 images means 90% agreement. Large classes contribute more images, so a high overall value can hide poor results on a small class.'],
    ['support', 'Samples (support)', 'The number of images with a particular reference label. A class with 100 samples contributes 100 images to the evaluation, regardless of what the model predicts. Counts describe image tiles, not patients, cells, or tissue area.'],
    ['precision', 'Precision', 'When the model predicts a particular class, how often does it match the reference? If it calls 12 images “smooth muscle” and 8 have that reference label, precision is 8 ÷ 12 = 66.7%. Low precision means the model includes many images from other classes.'],
    ['recall', 'Recall', 'Of all images with a particular reference label, how many does the model find? If there are 10 reference smooth-muscle images and the model identifies 8, recall is 8 ÷ 10 = 80%. Low recall means it misses many images from that class.'],
    ['f1', 'F1 score', 'A single score that combines precision and recall using their harmonic mean. It drops when either measure is low; one strong measure cannot fully compensate for a weak one. With precision 66.7% and recall 80%, F1 is about 72.7%. Formula: F1 = 2 × precision × recall ÷ (precision + recall).'],
    ['macro-f1', 'Macro F1', 'Calculate F1 separately for every class, then take the ordinary average. Each class has equal weight, whether it has 100 images or 1,000. This differs from overall label agreement, which counts every image equally.'],
    ['balanced-accuracy', 'Balanced accuracy', 'Calculate recall separately for every class, then average those recall values. Each class has equal weight. Here, balanced accuracy is the mean recall across all nine tissue categories; it is not the average of precision and recall.'],
    ['confusion-matrix', 'Confusion matrix', 'A table showing where predictions match or differ from the reference labels. Here, rows are reference classes and columns are predicted classes. Diagonal cells contain matches; other cells show which classes the model confuses. Row percentage divides a cell’s count by all images in that reference row.'],
    ['benchmark', 'Benchmark', 'A named dataset used to measure model behavior. The Benchmark page shows archived CONCH results on CRC-VAL-HE-7K. Those numbers do not measure the separate MobileNet model running in this browser, and neither result establishes accuracy on your own images.']
  ];
  const dialog = document.createElement('dialog');
  dialog.id = 'terms-dialog';
  dialog.className = 'terms-dialog';
  dialog.setAttribute('aria-labelledby', 'terms-title');
  const heading = document.createElement('div');
  heading.className = 'terms-heading';
  const title = document.createElement('h2'); title.id = 'terms-title'; title.textContent = 'Terms & metrics';
  const close = document.createElement('button'); close.type = 'button'; close.textContent = 'Close'; close.className = 'terms-close'; close.autofocus = true;
  close.addEventListener('click', () => dialog.close()); heading.append(title, close); dialog.append(heading);
  const intro = document.createElement('p'); intro.className = 'terms-intro'; intro.textContent = 'A plain-language guide to reading labels and results. The small number examples below are hypothetical—not additional benchmark results.'; dialog.append(intro);
  const sections = new Map();
  for (const [key, name, explanation] of entries) {
    const details = document.createElement('details'); details.dataset.metric = key;
    const summary = document.createElement('summary'); summary.textContent = name;
    const text = document.createElement('p'); text.textContent = explanation;
    details.append(summary, text); dialog.append(details); sections.set(key, details);
  }
  const note = document.createElement('p'); note.className = 'terms-note'; note.textContent = 'Use precision and recall together. A single high score does not establish clinical reliability or performance on a new collection.'; dialog.append(note);
  document.body.append(dialog);
  document.querySelectorAll('[data-term]').forEach(button => {
    button.setAttribute('aria-haspopup', 'dialog'); button.setAttribute('aria-controls', dialog.id);
    if (button.dataset.term) button.setAttribute('aria-label', 'Explain ' + sections.get(button.dataset.term)?.querySelector('summary').textContent);
    button.addEventListener('click', () => {
      sections.forEach(section => { section.open = false; });
      const target = sections.get(button.dataset.term);
      if (target) target.open = true;
      if (!dialog.open) dialog.showModal();
      dialog.scrollTop = 0;
      if (target) requestAnimationFrame(() => { target.querySelector('summary').focus({preventScroll:true}); target.scrollIntoView({block:'nearest'}); });
    });
  });
})();
