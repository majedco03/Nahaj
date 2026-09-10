<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import {
		createTask,
		getCourses,
		getPlan,
		getProgress,
		deleteTask,
		updateTask,
		type Course,
		type StudyTask,
		type TaskStatus,
		type Progress
	} from '$lib/apis/nahaj';

	let tasks: StudyTask[] = [];
	let courses: Course[] = [];
	let semester: any = null;
	let summary = { total: 0, done: 0, completion_percent: 0 };
	let progress: Progress | null = null;
	let loading = true;
	let filterCourse = '';
	let filterStatus: TaskStatus | 'all' = 'all';
	let editing: StudyTask | null = null;
	let showEditor = false;
	let form = { course_id: '', title: '', notes: '', start_at: '', end_at: '' };

	$: courseById = new Map(courses.map((course) => [course.id, course]));
	$: visibleTasks = tasks
		.filter((task) => !filterCourse || task.course_id === filterCourse)
		.filter((task) => filterStatus === 'all' || task.status === filterStatus)
		.sort((a, b) => new Date(a.start_at).getTime() - new Date(b.start_at).getTime());
	$: groupedTasks = visibleTasks.reduce((groups: Record<string, StudyTask[]>, task) => {
		const day = task.start_at.slice(0, 10);
		(groups[day] ??= []).push(task);
		return groups;
	}, {});

	function toLocalInput(value: string | Date) {
		const date = value instanceof Date ? value : new Date(value);
		const offset = date.getTimezoneOffset() * 60000;
		return new Date(date.getTime() - offset).toISOString().slice(0, 16);
	}

	function openPrompt(prompt: string) {
		window.localStorage.setItem('nahaj-chat-draft', prompt);
		goto('/');
	}

	async function load() {
		loading = true;
		try {
			const [plan, courseValues, progressValue] = await Promise.all([getPlan(), getCourses(), getProgress()]);
			semester = plan.semester;
			tasks = plan.tasks;
			summary = plan.summary;
			courses = courseValues;
			progress = progressValue;
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			loading = false;
		}
	}

	function newTask(start?: string) {
		editing = null;
		form = {
			course_id: filterCourse || '',
			title: '',
			notes: '',
			start_at: start ? toLocalInput(start) : toLocalInput(new Date()),
			end_at: start ? toLocalInput(new Date(new Date(start).getTime() + 60 * 60000)) : toLocalInput(new Date(Date.now() + 60 * 60000))
		};
		showEditor = true;
	}

	function editTask(task: StudyTask) {
		editing = task;
		form = { course_id: task.course_id ?? '', title: task.title, notes: task.notes ?? '', start_at: toLocalInput(task.start_at), end_at: toLocalInput(task.end_at) };
		showEditor = true;
	}

	async function saveTask() {
		if (!form.title.trim()) return;
		const payload = { course_id: form.course_id || null, title: form.title.trim(), notes: form.notes || null, start_at: new Date(form.start_at).toISOString(), end_at: new Date(form.end_at).toISOString() };
		try {
			if (editing) {
				const saved = await updateTask(editing.id, payload);
				tasks = tasks.map((task) => (task.id === saved.id ? saved : task));
			} else {
				tasks = [...tasks, await createTask(payload as any)];
			}
			showEditor = false;
			await load();
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	async function setStatus(task: StudyTask, status: TaskStatus) {
		try {
			const saved = await updateTask(task.id, { status });
			tasks = tasks.map((item) => (item.id === saved.id ? saved : item));
			const [plan, progressValue] = await Promise.all([getPlan(), getProgress()]);
			tasks = plan.tasks;
			summary = plan.summary;
			progress = progressValue;
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	async function removeTask(task: StudyTask) {
		if (!confirm(`Delete “${task.title}”?`)) return;
		try {
			await deleteTask(task.id);
			tasks = tasks.filter((item) => item.id !== task.id);
			await load();
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	onMount(load);
</script>

<svelte:head><title>Plan / Nahaj</title></svelte:head>

<div class="min-h-0 min-w-0 flex-1 overflow-y-auto px-4 py-8 @3xl:px-10">
	<div class="mx-auto max-w-6xl space-y-8">
		<div class="flex flex-wrap items-start justify-between gap-4">
			<div><p class="text-xs font-medium uppercase tracking-[0.22em] text-indigo-500">Nahaj</p><h1 class="mt-2 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">Study Plan</h1><p class="mt-2 text-sm text-gray-500 dark:text-gray-400">{semester ? `${semester.name} · ${semester.start_date} → ${semester.end_date}` : 'Set up a semester in Courses to start planning.'}</p></div>
			<div class="flex gap-2"><button class="secondary" on:click={() => openPrompt('Review my semester context and generate a realistic study plan. Ask for any missing deadlines or availability first.')}>Generate Plan</button><button class="primary" on:click={() => openPrompt('Regenerate my study plan using my current task completion and quiz progress. Keep existing tasks unless I explicitly ask to replace them.')}>Regenerate Plan</button></div>
		</div>

		<div class="grid gap-3 @2xl:grid-cols-3"><div class="card"><span class="label">Tasks completed</span><strong>{summary.done} / {summary.total}</strong><div class="meter"><span style={`width:${summary.completion_percent}%`}></span></div></div><div class="card"><span class="label">Overall completion</span><strong>{Math.round(summary.completion_percent)}%</strong><p class="muted">Tasks remain advisory until you edit them.</p></div><div class="card"><span class="label">Quiz average</span><strong>{progress?.quiz_average_percent == null ? '—' : `${Math.round(progress.quiz_average_percent)}%`}</strong><p class="muted">Based on completed attempts.</p></div></div>

		<div class="flex flex-wrap items-center gap-2"><select class="field" bind:value={filterCourse}><option value="">All courses</option>{#each courses as course}<option value={course.id}>{course.code} · {course.name}</option>{/each}</select><select class="field" bind:value={filterStatus}><option value="all">All statuses</option><option value="todo">To do</option><option value="done">Done</option><option value="skipped">Skipped</option></select><button class="primary ml-auto" on:click={() => newTask()}>New task</button></div>

		{#if loading}<div class="py-20 text-center text-sm text-gray-400">Loading your plan…</div>{:else if Object.keys(groupedTasks).length === 0}<div class="empty"><div class="text-4xl">🗓️</div><h2 class="mt-3 text-lg font-medium text-gray-900 dark:text-white">No tasks match this view</h2><p class="mt-1 text-sm text-gray-500">Create a task or ask the supervisor to shape your week.</p></div>{:else}<div class="space-y-5">{#each Object.entries(groupedTasks) as [day, dayTasks]}<section><div class="mb-2 flex items-center justify-between"><h2 class="text-sm font-semibold uppercase tracking-wide text-gray-500">{new Date(`${day}T00:00:00`).toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' })}</h2><button class="text-xs text-indigo-600 hover:underline" on:click={() => newTask(`${day}T09:00:00`)}>+ task</button></div><div class="space-y-2">{#each dayTasks as task (task.id)}<article class="task"><div class="flex items-start gap-3"><input type="checkbox" checked={task.status === 'done'} aria-label={`Mark ${task.title} done`} on:change={(event) => setStatus(task, (event.currentTarget as HTMLInputElement).checked ? 'done' : 'todo')} /><div class="min-w-0 flex-1"><div class="flex flex-wrap items-center gap-2"><h3 class="font-medium {task.status === 'done' ? 'line-through text-gray-400' : 'text-gray-900 dark:text-white'}">{task.title}</h3>{#if task.course_id}<span class="course-pill" style={`--course-color:${courseById.get(task.course_id)?.color ?? '#6366f1'}`}>{courseById.get(task.course_id)?.code ?? 'Course'}</span>{/if}<span class="status-pill">{task.status}</span></div><p class="mt-1 text-xs text-gray-500">{new Date(task.start_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })} – {new Date(task.end_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}{#if task.notes} · {task.notes}{/if}</p></div><div class="flex shrink-0 gap-2"><button class="icon-link" on:click={() => editTask(task)}>Edit</button><button class="icon-link text-red-500" on:click={() => removeTask(task)}>Delete</button></div></div></article>{/each}</div></section>{/each}</div>{/if}
	</div>
</div>

{#if showEditor}<div class="overlay" role="presentation" on:click={(event) => event.target === event.currentTarget && (showEditor = false)}><form class="modal" on:submit|preventDefault={saveTask}><div class="flex items-center justify-between"><h2 class="text-lg font-medium text-gray-900 dark:text-white">{editing ? 'Edit task' : 'New task'}</h2><button type="button" class="text-gray-400" on:click={() => (showEditor = false)}>✕</button></div><label>Course<select class="field mt-1 w-full" bind:value={form.course_id}><option value="">No course</option>{#each courses as course}<option value={course.id}>{course.code} · {course.name}</option>{/each}</select></label><label>Title<input class="field mt-1 w-full" bind:value={form.title} required /></label><label>Notes<textarea class="field mt-1 w-full" rows="3" bind:value={form.notes}></textarea></label><div class="grid gap-3 @2xl:grid-cols-2"><label>Starts<input class="field mt-1 w-full" type="datetime-local" bind:value={form.start_at} required /></label><label>Ends<input class="field mt-1 w-full" type="datetime-local" bind:value={form.end_at} required /></label></div><div class="flex justify-end gap-2"><button type="button" class="secondary" on:click={() => (showEditor = false)}>Cancel</button><button class="primary">Save task</button></div></form></div>{/if}
