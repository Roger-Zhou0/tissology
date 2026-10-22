'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const basePath = location.pathname.slice(0, location.pathname.lastIndexOf('/') + 1);
  const baseURL = new URL(basePath, location.origin);
  const fragments = new URLSearchParams(location.hash.slice(1));
  // Tokens are invitation/setup credentials. Keep them out of subsequent URLs,
  // persistent browser storage, network query strings, and copied page links.
  let invitationToken = fragments.get('invite') || '';
  let setupCode = fragments.get('setup') || '';
  if (fragments.has('invite') || fragments.has('setup')) {
    history.replaceState(null, '', location.pathname + location.search);
  }
  let session = null;
  let generation = 0;
  let issuedInvitation = null;
  let confirmation = null;
  let initialSessionRead = true;
  const pending = new Set();
  const endpoint = path => new URL('api/lab/' + path, baseURL).href;
  const appURL = path => new URL(path, baseURL).href;
  const publishSession = value => window.TissueLab?.acceptSession
    ? window.TissueLab.acceptSession(value)
    : window.dispatchEvent(new CustomEvent('tissue-lab-session',{detail:value}));
  const dateText = value => {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date.toLocaleString([], {dateStyle:'medium', timeStyle:'short'}) : 'Time unavailable';
  };
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };
  const errorMessage = error => error?.message || 'The request could not be completed. Try again.';
  const clearError = () => { $('lab-error').hidden = true; $('lab-error').textContent = ''; };
  const showError = error => {
    $('lab-error').textContent = errorMessage(error);
    $('lab-error').hidden = false;
    $('lab-error').focus();
  };
  const setStatus = text => { $('lab-status').textContent = text; };

  function safeNext() {
    const value = params.get('next');
    if (!value || /[\\\u0000-\u001f\u007f]/.test(value) || value.startsWith('//')) return appURL('workspace');
    try {
      const url = new URL(value, baseURL);
      if (url.origin !== location.origin || !url.pathname.startsWith(basePath)) return appURL('workspace');
      const route = url.pathname.slice(basePath.length);
      if (!/^(workspace|collections|semantic|research|report|classification|feedback|spatial)$/.test(route)
          && !/^feedback-studies\/[a-f0-9]{32}\/(?:workspace|classification|report)?$/.test(route)) return appURL('workspace');
      url.hash = '';
      return url.href;
    } catch (_) { return appURL('workspace'); }
  }

  async function request(path, options = {}) {
    const headers = {'Accept':'application/json'};
    const init = {method:options.method || 'GET', credentials:'same-origin', cache:'no-store', headers};
    if (options.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(options.body);
    }
    if (init.method !== 'GET') {
      headers['X-Lab-Request'] = '1';
      if (session?.csrf_token) headers['X-CSRF-Token'] = session.csrf_token;
    }
    let response;
    try { response = await fetch(endpoint(path), init); }
    catch (_) { throw new Error('Cannot reach the workspace. Check your connection, then try again.'); }
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error('The server returned an unexpected response. Refresh this page and try again.'); }
    if (!response.ok) {
      const detail = typeof data?.detail === 'string' ? data.detail : typeof data?.error === 'string' ? data.error : null;
      const error = new Error(detail || `The request could not be completed (${response.status}). Try again.`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  async function withBusy(key, control, label, action) {
    if (pending.has(key)) return;
    pending.add(key);
    const original = control?.textContent;
    if (control) { control.disabled = true; if (label) control.textContent = label; }
    clearError();
    try { await action(); }
    catch (error) {
      if (error.status === 401 && session?.authenticated && !['login','setup','join'].includes(key)) {
        session = {...session, authenticated:false, user:null, csrf_token:null};
        renderAuth('login');
        showError(new Error('Your session has ended. Sign in again to continue.'));
      } else showError(error);
    } finally {
      pending.delete(key);
      if (control) { control.disabled = false; control.textContent = original; }
    }
  }

  function hideSections() {
    for (const id of ['lab-login','lab-setup','lab-join','lab-team']) $(id).hidden = true;
    $('lab-retry').hidden = true;
    setStatus('');
  }

  function renderAuth(view) {
    generation++;
    if(session) publishSession(session);
    clearIssuedInvitation();
    for(const id of ['lab-members','lab-invitations','lab-activity-items']) $(id).replaceChildren();
    hideSections();
    document.body.dataset.labAuthenticated = 'false';
    if (session?.setup_required) view = 'setup';
    else if (view === 'setup') view = 'login';
    if (!['setup','join','login'].includes(view)) view = 'login';
    const name = session?.workspace?.name || 'your lab';
    $('lab-intro').textContent = 'Collections, searches, findings and studies are shared with everyone in your lab.';
    $(`lab-${view}`).hidden = false;
    if (view === 'setup') {
      $('lab-title').textContent = 'Set up your lab workspace';
      if (setupCode) { $('setup-code').value = setupCode; setupCode = ''; }
    } else if (view === 'join') {
      $('lab-title').textContent = `Join ${name}`;
      $('lab-join-form').hidden = !invitationToken;
      $('lab-join-note').textContent = invitationToken
        ? 'Create your account using the email address on your invitation.'
        : 'Open the complete invitation link your lab owner shared with you. If you refreshed this page, reopen that link.';
    } else $('lab-title').textContent = session?.workspace?.name ? `Sign in to ${name}` : 'Sign in to your lab';
    document.title = $('lab-title').textContent + ' · Tissology';
    const signin = new URL('lab', baseURL);
    signin.searchParams.set('view','login');
    if (params.get('next')) signin.searchParams.set('next', new URL(safeNext()).pathname + new URL(safeNext()).search);
    $('lab-signin-link').href = signin.href;
  }

  function renderTeam() {
    hideSections();
    document.body.dataset.labAuthenticated = 'true';
    $('lab-team').hidden = false;
    $('lab-title').textContent = session.workspace?.name || 'Your lab workspace';
    $('lab-intro').textContent = 'Everyone in this lab can view and edit the shared collections, searches, findings and studies.';
    document.title = $('lab-title').textContent + ' · Tissology';
    $('lab-current-name').textContent = `${session.user.name} · ${session.user.role === 'owner' ? 'Lab owner' : 'Lab member'}`;
    $('lab-current-email').textContent = session.user.email;
    $('lab-open-workspace').href = appURL('workspace');
    $('lab-owner-tools').hidden = session.user.role !== 'owner';
    if (invitationToken) setStatus('You are already signed in. Sign out first if you want to join with a different account.');
    loadTeam();
  }

  function renderMembers(data) {
    const list = $('lab-members'); list.replaceChildren();
    const members = Array.isArray(data.members) ? data.members : [];
    $('lab-members-status').textContent = members.length ? `${members.length} ${members.length === 1 ? 'member' : 'members'}` : 'No members were returned. Refresh to try again.';
    for (const member of members) {
      const row = element('li');
      const person = element('div', undefined, 'lab-person');
      const name = element('p', member.name || member.email, 'lab-person-name');
      const badge = element('span', member.id === session.user.id ? (member.role === 'owner' ? 'Owner · You' : 'You') : member.role === 'owner' ? 'Owner' : 'Member', 'lab-person-role');
      name.append(badge);
      person.append(name, element('p', member.email, 'lab-person-meta'));
      row.append(person);
      if (session.user.role === 'owner' && member.id !== session.user.id) {
        const remove = element('button','Remove'); remove.type='button';
        remove.setAttribute('aria-label',`Remove ${member.name || member.email} from lab`);
        remove.addEventListener('click', () => confirmAction({
          title:'Remove lab access?',
          detail:`${member.name || member.email} will be signed out and will no longer have access. Their existing contributions will stay in the shared workspace.`,
          label:'Remove access', key:`member-${member.id}`, origin:remove,
          action:async () => { await request('members/' + encodeURIComponent(member.id), {method:'DELETE'}); await loadTeam(); setStatus('Lab access removed. Shared work has been kept.'); }
        }));
        row.append(remove);
      }
      list.append(row);
    }
    if (session.user.role === 'owner') renderInvitations(data.invitations || []);
  }

  function renderInvitations(invitations) {
    const list=$('lab-invitations'); list.replaceChildren();
    $('lab-invitations-status').textContent = invitations.length ? '' : 'No pending invitations.';
    for (const invitation of invitations) {
      const row=element('li');
      const person=element('div', undefined, 'lab-person');
      person.append(element('p',invitation.email,'lab-person-name'), element('p',`Expires ${dateText(invitation.expires_at)}`,'lab-person-meta'));
      const revoke=element('button','Revoke'); revoke.type='button';
      revoke.setAttribute('aria-label',`Revoke invitation for ${invitation.email}`);
      revoke.addEventListener('click',()=>confirmAction({
        title:'Revoke this invitation?', detail:`The invitation link for ${invitation.email} will stop working. You can create a new one later.`,
        label:'Revoke invitation', key:`invitation-${invitation.id}`, origin:revoke,
        action:async()=>{
          await request('invitations/' + encodeURIComponent(invitation.id), {method:'DELETE'});
          if(issuedInvitation?.id===invitation.id) clearIssuedInvitation();
          await loadTeam(); setStatus('Invitation revoked.');
        }
      }));
      row.append(person,revoke);list.append(row);
    }
  }

  function renderActivity(data) {
    const items=Array.isArray(data.items) ? data.items : [];
    const list=$('lab-activity-items');list.replaceChildren();
    $('lab-activity-status').textContent=items.length ? '' : 'No account activity recorded yet.';
    for(const item of items.slice(0,30)) {
      const row=element('li');
      const line=element('p');
      line.append(element('strong',item.actor_name || 'Lab account'),document.createTextNode(' · ' + (item.action || 'Workspace action')));
      if (item.status === null || item.status === undefined) line.append(document.createTextNode(' · Outcome unconfirmed'));
      else if (Number(item.status) >= 400) line.append(document.createTextNode(' · Not completed'));
      const time=element('time',dateText(item.created_at));
      if(Number.isFinite(new Date(item.created_at).getTime())) time.dateTime=new Date(item.created_at).toISOString();
      row.append(line,time);list.append(row);
    }
  }

  async function loadTeam() {
    if(!session?.authenticated) return;
    const current=++generation;
    $('lab-refresh').disabled=true;
    $('lab-members-status').textContent='Loading members…';
    $('lab-activity-status').textContent='Loading activity…';
    const results=await Promise.allSettled([request('members'),request('activity')]);
    if(current!==generation || !session?.authenticated) return;
    $('lab-refresh').disabled=false;
    const unauthorized=results.find(result=>result.status==='rejected' && result.reason.status===401);
    if(unauthorized) {
      session={...session,authenticated:false,user:null,csrf_token:null};
      renderAuth('login');showError(new Error('Your session has ended. Sign in again to continue.'));return;
    }
    if(results[0].status==='fulfilled') renderMembers(results[0].value);
    else { $('lab-members-status').textContent='Members could not be refreshed. Use Refresh to try again.'; showError(results[0].reason); }
    if(results[1].status==='fulfilled') renderActivity(results[1].value);
    else $('lab-activity-status').textContent='Activity could not be loaded. Use Refresh to try again.';
  }

  function confirmAction(details) {
    confirmation=details;
    $('lab-confirm-title').textContent=details.title;
    $('lab-confirm-detail').textContent=details.detail;
    $('lab-confirm-action').textContent=details.label;
    $('lab-confirm-action').disabled=false;
    $('lab-confirm').showModal();
  }
  $('lab-confirm-cancel').addEventListener('click',()=> $('lab-confirm').close());
  $('lab-confirm').addEventListener('close',()=> { confirmation=null; });
  $('lab-confirm-action').addEventListener('click',()=>{
    const details=confirmation;if(!details) return;
    $('lab-confirm').close();
    withBusy(details.key,details.origin,'Updating…',details.action);
  });

  function clearIssuedInvitation() {
    issuedInvitation=null;
    $('lab-invite-created').hidden=true;
    $('lab-invite-url').value='';
    $('lab-invite-copy-status').textContent='';
    $('lab-invite-created-note').textContent='';
  }

  for(const view of ['login','setup','join']) {
    $(`lab-${view}-form`).addEventListener('submit',event=>{
      event.preventDefault();
      const form=event.currentTarget;
      if(!form.reportValidity()) return;
      const body=Object.fromEntries(new FormData(form));
      for(const key of ['name','email','workspace_name','code']) if(typeof body[key]==='string') body[key]=body[key].trim();
      if(view==='join') body.token=invitationToken;
      withBusy(view,form.querySelector('button[type=submit]'),view==='login'?'Signing in…':view==='setup'?'Creating workspace…':'Joining…',async()=>{
        const result=await request(view,{method:'POST',body});
        session=result.session || result;
        if(!session.authenticated) session=await request('session');
        if(!session.authenticated || !session.user) throw new Error('Your session could not be opened. Refresh this page and sign in.');
        publishSession(session);
        form.reset();invitationToken='';setupCode='';
        if(view==='setup') {
          const url=new URL('lab',baseURL);url.searchParams.set('view','team');history.replaceState(null,'',url.href);
          renderTeam();setStatus('Your lab is ready. Invite a collaborator or open the workspace.');
        } else location.assign(safeNext());
      });
    });
  }

  $('lab-invite-form').addEventListener('submit',event=>{
    event.preventDefault();const form=event.currentTarget;if(!form.reportValidity()) return;
    withBusy('invite',form.querySelector('button[type=submit]'),'Creating…',async()=>{
      const invitation=await request('invitations',{method:'POST',body:{email:$('invite-email').value.trim()}});
      if(!invitation.token) throw new Error('The invitation response did not include a link. Refresh the pending invitations before creating another.');
      issuedInvitation={id:invitation.id,email:invitation.email};
      const link=new URL('lab',baseURL);link.searchParams.set('view','join');link.hash=new URLSearchParams({invite:invitation.token}).toString();
      $('lab-invite-url').value=link.href;
      $('lab-invite-copy-status').textContent='';
      $('lab-invite-created-note').textContent=`For ${invitation.email}. Expires ${dateText(invitation.expires_at)}. Email is not sent automatically; share this link directly. The link is shown only now.`;
      $('lab-invite-created').hidden=false;form.reset();
      await loadTeam();setStatus('Invitation created. Copy the link to share it.');
      $('lab-invite-url').focus();
    });
  });
  $('lab-copy-invite').addEventListener('click',()=>withBusy('copy',$('lab-copy-invite'),'Copying…',async()=>{
    const input=$('lab-invite-url');
    try {
      if(!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(input.value);
      $('lab-invite-copy-status').textContent='Link copied.';
    } catch(_) {
      input.focus();input.select();
      $('lab-invite-copy-status').textContent='Link selected. Copy it using your browser or keyboard.';
    }
  }));
  $('lab-logout').addEventListener('click',()=>withBusy('logout',$('lab-logout'),'Signing out…',async()=>{
    await request('logout',{method:'POST'});
    clearIssuedInvitation();
    publishSession({...session,authenticated:false,user:null,csrf_token:null});
    location.assign(appURL('lab?view=login'));
  }));
  $('lab-refresh').addEventListener('click',()=>{clearError();loadTeam();});

  async function start() {
    clearError();hideSections();setStatus('Checking your session…');
    try {
      const read = initialSessionRead && window.TissueLab?.ready ? window.TissueLab.ready : request('session');
      initialSessionRead = false;
      session=await read;
      publishSession(session);
      if(session.authenticated && session.user) renderTeam();
      else renderAuth(params.get('view') || (invitationToken?'join':'login'));
    } catch(error) {
      $('lab-title').textContent='Connect to your lab';setStatus('');showError(error);$('lab-retry').hidden=false;
    }
  }
  $('lab-retry').addEventListener('click',start);
  start();
})();
