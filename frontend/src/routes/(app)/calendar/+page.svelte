<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import {
		createTask,
		deleteTask,
		getCourses,
		getPlan,
		updateTask,
		type Course,
		type StudyTask
	} from '$lib/apis/nahaj';

	let view: 'month' | 'week' | 'day' = 'month';
	let currentDate = new Date();
	let tasks: StudyTask[] = [];
	let courses: Course[] = [];
	let loading = true;
	let showEditor = false;
	let editing: StudyTask | null = null;
	let draggedTaskId = '';
	let form = { course_id: '', title: '', notes: '', start_at: '', end_at: '' };

	$: courseById = new Map(courses.map((course) => [course.id, course]));
	$: rangeDays = getRangeDays();
	$: headerText = view === 'day' ? formatLong(currentDate) : view === 'week' ? `${formatShort(rangeDays[0])} – ${formatShort(rangeDays[rangeDays.length - 1])}` : currentDate.toLocaleDateString(undefined, { month: 'long', year: 'numeric' });

	function dayKey(date: Date) { return date.toISOString().slice(0, 10); }
	function formatShort(date: Date) { return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }); }
	function formatLong(date: Date) { return date.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }); }

	function getRangeDays() {
		const anchor = new Date(currentDate);
		anchor.setHours(0, 0, 0, 0);
		if (view === 'day') return [anchor];
		if (view === 'week') {
			const start = new Date(anchor);
			start.setDate(start.getDate() - start.getDay());
			return Array.from({ length: 7 }, (_, index) => { const date = new Date(start); date.setDate(start.getDate() + index); return date; });
		}
		const start = new Date(anchor.getFullYear(), anchor.getMonth(), 1);
		start.setDate(start.getDate() - start.getDay());
		return Array.from({ length: 42 }, (_, index) => { const date = new Date(start); date.setDate(start.getDate() + index); return date; });
	}

	function tasksFor(date: Date) { return tasks.filter((task) => dayKey(new Date(task.start_at)) === dayKey(date)); }

	async function load() {
		loading = true;
		try {
			const days = getRangeDays();
			const from = new Date(days[0]);
			const to = new Date(days[days.length - 1]);
			to.setHours(23, 59, 59, 999);
			const [plan, courseValues] = await Promise.all([getPlan({ from: from.toISOString(), to: to.toISOString() }), getCourses()]);
			tasks = plan.tasks;
			courses = courseValues;
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			loading = false;
		}
	}

	function move(delta: number) {
		const next = new Date(currentDate);
		if (view === 'month') next.setMonth(next.getMonth() + delta);
		else if (view === 'week') next.setDate(next.getDate() + delta * 7);
		else next.setDate(next.getDate() + delta);
		currentDate = next;
		load();
	}

	function today() { currentDate = new Date(); load(); }

	function toLocalInput(value: string | Date) {
		const date = value instanceof Date ? value : new Date(value);
		const offset = date.getTimezoneOffset() * 60000;
		return new Date(date.getTime() - offset).toISOString().slice(0, 16);
	}

	function openEditor(date?: Date, task?: StudyTask) {
		editing = task ?? null;
		if (task) {
			form = { course_id: task.course_id ?? '', title: task.title, notes: task.notes ?? '', start_at: toLocalInput(task.start_at), end_at: toLocalInput(task.end_at) };
		} else {
			const start = date ? new Date(date) : new Date();
			start.setHours(9, 0, 0, 0);
			const end = new Date(start); end.setHours(end.getHours() + 1);
			form = { course_id: '', title: '', notes: '', start_at: toLocalInput(start), end_at: toLocalInput(end) };
		}
		showEditor = true;
	}

	async function save() {
		if (!form.title.trim()) return;
		const payload = { course_id: form.course_id || null, title: form.title.trim(), notes: form.notes || null, start_at: new Date(form.start_at).toISOString(), end_at: new Date(form.end_at).toISOString() };
		try {
			if (editing) await updateTask(editing.id, payload);
			else await createTask(payload as any);
			showEditor = false;
			await load();
		} catch (err) { toast.error(`${err}`); }
	}

	async function remove(task: StudyTask) {
		if (!confirm(`Delete “${task.title}”?`)) return;
		try { await deleteTask(task.id); await load(); } catch (err) { toast.error(`${err}`); }
	}

	async function removeEditing() {
		if (!editing) return;
		const task = editing;
		showEditor = false;
		await remove(task);
	}

	async function dropOn(date: Date) {
		if (!draggedTaskId) return;
		const task = tasks.find((item) => item.id === draggedTaskId);
		draggedTaskId = '';
		if (!task) return;
		const original = new Date(task.start_at);
		const start = new Date(date);
		start.setHours(original.getHours(), original.getMinutes(), 0, 0);
		const duration = new Date(task.end_at).getTime() - original.getTime();
		try { await updateTask(task.id, { start_at: start.toISOString(), end_at: new Date(start.getTime() + duration).toISOString() }); await load(); } catch (err) { toast.error(`${err}`); }
	}

	onMount(load);
</script>

<svelte:head><title>Calendar / Nahaj</title></svelte:head>

<div class="min-h-0 min-w-0 flex-1 overflow-y-auto px-3 py-6 @3xl:px-8">
	<div class="mx-auto max-w-7xl space-y-5">
		<div class="flex flex-wrap items-center justify-between gap-3"><div><p class="text-xs font-medium uppercase tracking-[0.22em] text-indigo-500">Nahaj</p><h1 class="mt-1 text-2xl font-semibold text-gray-900 dark:text-white">Calendar</h1></div><div class="flex items-center gap-1"><button class="secondary" on:click={() => move(-1)} aria-label="Previous">‹</button><button class="secondary" on:click={today}>Today</button><button class="secondary" on:click={() => move(1)} aria-label="Next">›</button><button class="primary ml-2" on:click={() => openEditor()}>New task</button></div></div>
		<div class="flex flex-wrap items-center justify-between gap-3"><h2 class="text-lg font-medium text-gray-800 dark:text-gray-200">{headerText}</h2><div class="segmented">{#each ['month', 'week', 'day'] as option}<button class={view === option ? 'segment-active' : ''} on:click={() => { view = option as typeof view; load(); }}>{option[0].toUpperCase() + option.slice(1)}</button>{/each}</div></div>
		{#if loading}<div class="py-20 text-center text-sm text-gray-400">Loading calendar…</div>{:else if view === 'month'}<div class="calendar-grid month-grid">{#each ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as name}<div class="day-name">{name}</div>{/each}{#each rangeDays as date}<div role="button" tabindex="0" aria-label={`Add task on ${formatLong(date)}`} class="day-cell {date.getMonth() === currentDate.getMonth() ? '' : 'day-muted'}" on:click={() => openEditor(date)} on:keydown={(event) => (event.key === 'Enter' || event.key === ' ') && openEditor(date)} on:dragover|preventDefault on:drop|preventDefault={() => dropOn(date)}><div class="flex items-center justify-between"><span class="day-number {dayKey(date) === dayKey(new Date()) ? 'today-number' : ''}">{date.getDate()}</span>{#if tasksFor(date).length}<span class="text-[0.65rem] text-gray-400">{tasksFor(date).length}</span>{/if}</div><div class="mt-1 space-y-1">{#each tasksFor(date) as task (task.id)}<button draggable="true" class="event" style={`--course-color:${courseById.get(task.course_id ?? '')?.color ?? '#6366f1'}`} on:dragstart={() => (draggedTaskId = task.id)} on:click|stopPropagation={() => openEditor(date, task)}><span class="truncate">{task.title}</span></button>{/each}</div></div>{/each}</div>{:else}<div class="calendar-grid {view === 'day' ? 'day-layout' : 'week-layout'}">{#each rangeDays as date}<section class="column"><div class="column-head"><span class="day-name">{date.toLocaleDateString(undefined, { weekday: 'short' })}</span><b class="day-number {dayKey(date) === dayKey(new Date()) ? 'today-number' : ''}">{date.getDate()}</b></div><button class="add-slot" on:click={() => openEditor(date)}>+ Add task</button><div class="space-y-2">{#each tasksFor(date) as task (task.id)}<button class="event block w-full text-left" style={`--course-color:${courseById.get(task.course_id ?? '')?.color ?? '#6366f1'}`} on:click={() => openEditor(date, task)}><span class="block truncate">{task.title}</span><span class="mt-1 block text-[0.65rem] opacity-70">{new Date(task.start_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span></button>{/each}</div></section>{/each}</div>{/if}
	</div>
</div>

{#if showEditor}<div class="overlay" role="presentation" on:click={(event) => event.target === event.currentTarget && (showEditor = false)}><form class="modal" on:submit|preventDefault={save}><div class="flex items-center justify-between"><h2 class="text-lg font-medium text-gray-900 dark:text-white">{editing ? 'Edit task' : 'New calendar task'}</h2><button type="button" class="text-gray-400" on:click={() => (showEditor = false)}>✕</button></div><label>Course<select class="field mt-1 w-full" bind:value={form.course_id}><option value="">No course</option>{#each courses as course}<option value={course.id}>{course.code} · {course.name}</option>{/each}</select></label><label>Title<input class="field mt-1 w-full" bind:value={form.title} required /></label><label>Notes<textarea class="field mt-1 w-full" rows="3" bind:value={form.notes}></textarea></label><div class="grid gap-3 @2xl:grid-cols-2"><label>Starts<input class="field mt-1 w-full" type="datetime-local" bind:value={form.start_at} required /></label><label>Ends<input class="field mt-1 w-full" type="datetime-local" bind:value={form.end_at} required /></label></div><div class="flex justify-end gap-2"><button type="button" class="secondary" on:click={() => (showEditor = false)}>Cancel</button>{#if editing}<button type="button" class="secondary text-red-500" on:click={removeEditing}>Delete</button>{/if}<button class="primary">Save task</button></div></form></div>{/if}
