'use strict';
(() => {
  const script = new URL(document.currentScript.src);
  const base = new URL('../', script);
  const nested=base.pathname.indexOf('/feedback-studies/');
  if(nested>=0) base.pathname=base.pathname.slice(0,nested+1);
  const labPage = location.pathname === base.pathname+'lab';
  const nativeFetch = window.fetch.bind(window);
  const marker = 'tissue-lab-member:'+base.pathname;
  let remembered = '';
  try { remembered = localStorage.getItem(marker) || ''; } catch (_) {}
  let session = null;
  let redirecting = false;
  const signin = () => {
    if(labPage || redirecting) return;
    redirecting=true;
    const target=new URL('lab',base);
    target.searchParams.set('view','login');
    target.searchParams.set('next',location.pathname.slice(base.pathname.length)+location.search);
    location.replace(target.href);
  };
  function acceptSession(value) {
    session=value;
    if(!value?.configured) {
      const previous=remembered;remembered='';
      let cleared=false;
      try {localStorage.removeItem(marker);cleared=localStorage.getItem(marker)===null;} catch (_) {}
      window.dispatchEvent(new CustomEvent('tissue-lab-session',{detail:value}));
      if(previous&&cleared&&!labPage) location.reload();
      return value;
    }
    const member=value.authenticated ? value.user.id : '';
    let stored=false;
    try {localStorage.setItem(marker,member);stored=localStorage.getItem(marker)===member;} catch (_) {}
    // A page rendered under another member must restart before it can save.
    if(!labPage && stored && remembered!==member && value.authenticated) {
      remembered=member;location.reload();return value;
    }
    remembered=member;
    window.dispatchEvent(new CustomEvent('tissue-lab-session',{detail:value}));
    return value;
  }
  const ready=nativeFetch(new URL('api/lab/session',base),{credentials:'same-origin',cache:'no-store'})
    .then(async response=>{
      if(!response.ok) throw new Error('The account service is unavailable. Reload before saving.');
      const current=acceptSession(await response.json());
      if(current.configured&&!current.authenticated) signin();
      return current;
    });
  // Prevent an unhandled rejection when a page only displays a connection error.
  ready.catch(()=>{});
  window.TissueLab={ready,acceptSession,storageKey:key=>remembered ? key+':member:'+remembered : key,
                   memberId:()=>remembered,session:()=>session};
  window.fetch=async (input,options={})=>{
    const url=new URL(typeof input==='string'||input instanceof URL ? input : input.url,document.baseURI);
    if(url.origin!==base.origin || !url.pathname.startsWith(base.pathname) || url.pathname.startsWith(base.pathname+'api/lab/')) return nativeFetch(input,options);
    const current=await ready;
    const method=(options.method||(input instanceof Request?input.method:'GET')).toUpperCase();
    const headers=new Headers(options.headers||(input instanceof Request?input.headers:undefined));
    if(current?.configured && session?.authenticated && !['GET','HEAD','OPTIONS'].includes(method)) headers.set('X-CSRF-Token',session.csrf_token);
    const response=await nativeFetch(input,{...options,headers});
    if(current?.configured && response.status===401) signin();
    return response;
  };
  window.addEventListener('storage',event=>{
    if(event.key===marker && event.newValue!==remembered && !labPage) location.reload();
  });
  window.addEventListener('pageshow',event=>{if(event.persisted) location.reload();});
})();
