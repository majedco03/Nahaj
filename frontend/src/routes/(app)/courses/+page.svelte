<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		createCourse,
		deleteCourse,
		deleteDocument,
		getCourses,
		getDocuments,
		getSemester,
		retryDocument,
		saveSemester,
		uploadDocument,
		updateCourse,
		documentContentUrl,
		type Collection,
		type Course,
		type DocumentRecord,
		type Semester
	} from '$lib/apis/nahaj';

	let semester: Semester | null = null;
	let courses: Course[] = [];
	let selectedCourseId = '';
	let collection: Collection = 'slides';
	let documents: DocumentRecord[] = [];
	let loading = true;
	let busy = false;
	let error = '';
	let uploadInput: HTMLInputElement;
	let selectedCourse: Course | null = null;

	let semesterForm = {
		name: 'Semester',
		start_date: new Date().toISOString().slice(0, 10),
		end_date: new Date(Date.now() + 120 * 86400000).toISOString().slice(0, 10),
		timezone: 'Asia/Riyadh'
	};
	let courseForm = { code: '', name: '', color: '#5b5ce2' };
	let editingCourseId = '';
	let editForm = { code: '', name: '', color: '' };

	$: selectedCourse = courses.find((course) => course.id === selectedCourseId) ?? null;

	async function loadWorkspace() {
		loading = true;
		error = '';
		try {
			const [semesterValue, courseValues] = await Promise.all([getSemester(), getCourses()]);
			semester = semesterValue;
			courses = courseValues;
			if (semester) {
				semesterForm = {
					name: semester.name,
					start_date: semester.start_date,
					end_date: semester.end_date,
					timezone: semester.timezone
				};
			}
			if (!selectedCourseId || !courses.some((course) => course.id === selectedCourseId)) {
				selectedCourseId = courses[0]?.id ?? '';
			}
			await loadDocuments();
		} catch (err) {
			error = `${err}`;
		} finally {
			loading = false;
		}
	}

	async function loadDocuments() {
		if (!selectedCourseId) {
			documents = [];
			return;
		}
		documents = await getDocuments(selectedCourseId, collection);
	}

	async function selectCourse(id: string) {
		selectedCourseId = id;
		editingCourseId = '';
		await loadDocuments();
	}

	async function switchCollection(next: Collection) {
		collection = next;
		await loadDocuments();
	}

	async function submitSemester() {
		busy = true;
		try {
			semester = await saveSemester(semesterForm);
			toast.success('Semester saved');
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			busy = false;
		}
	}

	async function submitCourse() {
		if (!courseForm.code.trim() || !courseForm.name.trim()) return;
		busy = true;
		try {
			const created = await createCourse({
				code: courseForm.code.trim(),
				name: courseForm.name.trim(),
				color: courseForm.color
			});
			courses = [...courses, created];
			selectedCourseId = created.id;
			courseForm = { code: '', name: '', color: '#5b5ce2' };
			await loadDocuments();
			toast.success('Course created');
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			busy = false;
		}
	}

	function beginEdit(course: Course) {
		editingCourseId = course.id;
		editForm = { code: course.code, name: course.name, color: course.color };
	}

	async function saveCourse() {
		if (!editingCourseId) return;
		try {
			const updated = await updateCourse(editingCourseId, editForm);
			courses = courses.map((course) => (course.id === updated.id ? updated : course));
			editingCourseId = '';
			toast.success('Course updated');
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	async function removeCourse(course: Course) {
		if (!confirm(`Delete ${course.code}? Remove its documents, tasks, and quizzes first.`)) return;
		try {
			await deleteCourse(course.id);
			courses = courses.filter((item) => item.id !== course.id);
			selectedCourseId = courses[0]?.id ?? '';
			await loadDocuments();
			toast.success('Course deleted');
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	async function handleUpload(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (!file || !selectedCourseId) return;
		try {
			await uploadDocument(selectedCourseId, collection, file);
			toast.success(`${file.name} is processing`);
			await loadDocuments();
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			if (uploadInput) uploadInput.value = '';
		}
	}

	async function retry(id: string) {
		try {
			await retryDocument(id);
			await loadDocuments();
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	async function removeDocument(document: DocumentRecord) {
		if (!confirm(`Delete ${document.filename}?`)) return;
		try {
			await deleteDocument(document.id);
			await loadDocuments();
		} catch (err) {
			toast.error(`${err}`);
		}
	}

	onMount(() => {
		loadWorkspace();
		const timer = window.setInterval(async () => {
			if (selectedCourseId && documents.some((document) => document.status === 'processing')) {
				await loadDocuments().catch(() => undefined);
			}
		}, 2500);
		return () => window.clearInterval(timer);
	});
</script>

<svelte:head><title>Courses / Nahaj</title></svelte:head>

<div class="min-h-0 min-w-0 flex-1 overflow-y-auto px-4 py-8 @md:px-10">
	<div class="mx-auto max-w-6xl space-y-8">
		<div>
			<p class="text-xs font-medium uppercase tracking-[0.22em] text-indigo-500">Nahaj</p>
			<h1 class="mt-2 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">Course Library</h1>
			<p class="mt-2 max-w-2xl text-sm text-gray-500 dark:text-gray-400">
				Keep each course's Slides and Past Exams together. Original files stay in Nahaj storage while your supervisor handles ingestion.
			</p>
		</div>

		{#if error}
			<div class="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300">{error}</div>
		{/if}

		{#if !semester && !loading}
			<section class="rounded-2xl border border-indigo-200 bg-indigo-50/70 p-5 dark:border-indigo-900/60 dark:bg-indigo-950/20">
				<h2 class="text-lg font-medium text-gray-900 dark:text-white">Set up your active semester</h2>
				<p class="mt-1 text-sm text-gray-600 dark:text-gray-400">There is one active semester at a time. You can update these dates later.</p>
				<form class="mt-4 grid gap-3 @md:grid-cols-4" on:submit|preventDefault={submitSemester}>
					<input class="field" bind:value={semesterForm.name} placeholder="Semester name" required />
					<input class="field" type="date" bind:value={semesterForm.start_date} required />
					<input class="field" type="date" bind:value={semesterForm.end_date} required />
					<button class="primary" disabled={busy}>Save semester</button>
				</form>
			</section>
		{:else if semester}
			<section class="rounded-2xl border border-gray-200 bg-white/70 p-5 shadow-sm dark:border-gray-800 dark:bg-gray-900/50">
				<div class="flex flex-wrap items-start justify-between gap-4">
					<div>
						<h2 class="text-lg font-medium text-gray-900 dark:text-white">{semester.name}</h2>
						<p class="mt-1 text-sm text-gray-500 dark:text-gray-400">{semester.start_date} → {semester.end_date} · {semester.timezone}</p>
					</div>
					<form class="flex flex-wrap items-center gap-2" on:submit|preventDefault={submitSemester}>
						<input class="field w-40" bind:value={semesterForm.name} aria-label="Semester name" />
						<input class="field" type="date" bind:value={semesterForm.start_date} aria-label="Semester start" />
						<input class="field" type="date" bind:value={semesterForm.end_date} aria-label="Semester end" />
						<button class="secondary" disabled={busy}>Update</button>
					</form>
				</div>
			</section>
		{/if}

		<section class="grid gap-5 @lg:grid-cols-[19rem_1fr]">
			<div class="space-y-4">
				<div class="rounded-2xl border border-gray-200 bg-white/70 p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/50">
					<div class="flex items-center justify-between"><h2 class="font-medium text-gray-900 dark:text-white">Courses</h2><span class="text-xs text-gray-400">{courses.length}</span></div>
					<form class="mt-4 space-y-2" on:submit|preventDefault={submitCourse}>
						<input class="field w-full" bind:value={courseForm.code} placeholder="Code (e.g. CS101)" disabled={!semester} required />
						<input class="field w-full" bind:value={courseForm.name} placeholder="Course name" disabled={!semester} required />
						<div class="flex gap-2"><input class="h-9 w-12 cursor-pointer rounded border border-gray-200 bg-transparent" type="color" bind:value={courseForm.color} disabled={!semester} /><button class="primary flex-1" disabled={!semester || busy}>Add course</button></div>
					</form>
					<div class="mt-4 space-y-1">
						{#each courses as course (course.id)}
							<button class="course-row {course.id === selectedCourseId ? 'course-row-active' : ''}" on:click={() => selectCourse(course.id)}>
								<span class="h-3 w-3 shrink-0 rounded-full" style={`background:${course.color}`}></span><span class="min-w-0 flex-1 truncate text-left"><b>{course.code}</b><span class="ml-2 text-gray-500">{course.name}</span></span>
							</button>
						{/each}
						{#if courses.length === 0}<p class="py-5 text-center text-sm text-gray-400">Add your first course.</p>{/if}
					</div>
				</div>
				{#if selectedCourse}
					<div class="rounded-2xl border border-gray-200 bg-white/70 p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/50">
						<div class="flex items-center justify-between"><h3 class="font-medium text-gray-900 dark:text-white">Course details</h3><button class="text-xs text-red-500" on:click={() => removeCourse(selectedCourse)}>Delete</button></div>
						{#if editingCourseId === selectedCourse.id}
							<div class="mt-3 space-y-2"><input class="field w-full" bind:value={editForm.code} /><input class="field w-full" bind:value={editForm.name} /><input class="h-9 w-12 cursor-pointer rounded border border-gray-200 bg-transparent" type="color" bind:value={editForm.color} /><div class="flex gap-2"><button class="primary flex-1" on:click={saveCourse}>Save</button><button class="secondary" on:click={() => (editingCourseId = '')}>Cancel</button></div></div>
			{:else}
				<div class="mt-3 flex items-center justify-between text-sm"><span>{selectedCourse.code} · {selectedCourse.name}</span><button class="secondary" on:click={() => beginEdit(selectedCourse)}>Edit</button></div>
			{/if}
					</div>
				{/if}
			</div>

			<div class="rounded-2xl border border-gray-200 bg-white/70 p-5 shadow-sm dark:border-gray-800 dark:bg-gray-900/50">
				{#if selectedCourse}
					<div class="flex flex-wrap items-center justify-between gap-3"><div><h2 class="text-xl font-medium text-gray-900 dark:text-white">{selectedCourse.code}: {selectedCourse.name}</h2><p class="mt-1 text-sm text-gray-500">Original course material</p></div><label class="primary cursor-pointer"><input bind:this={uploadInput} class="hidden" type="file" accept=".pdf,.pptx,.docx,.png,.jpg,.jpeg" on:change={handleUpload} />Upload file</label></div>
					<div class="mt-6 flex gap-1 border-b border-gray-200 dark:border-gray-800"><button class="tab {collection === 'slides' ? 'tab-active' : ''}" on:click={() => switchCollection('slides')}>Slides</button><button class="tab {collection === 'past_exam' ? 'tab-active' : ''}" on:click={() => switchCollection('past_exam')}>Past Exams</button></div>
					<div class="mt-4 overflow-x-auto"><table class="w-full min-w-[38rem] text-left text-sm"><thead class="text-xs uppercase tracking-wide text-gray-400"><tr><th class="py-2">File</th><th>Status</th><th>Added</th><th class="text-right">Actions</th></tr></thead><tbody>
						{#each documents as document (document.id)}<tr class="border-t border-gray-100 dark:border-gray-800"><td class="max-w-[18rem] truncate py-3 pr-3 text-gray-800 dark:text-gray-200">{document.filename}<div class="text-xs text-gray-400">{Math.ceil(document.size_bytes / 1024)} KB</div></td><td><span class="status status-{document.status}">{document.status}</span>{#if document.error}<div class="mt-1 max-w-[14rem] text-xs text-red-500">{document.error}</div>{/if}</td><td class="text-gray-500">{new Date(document.created_at).toLocaleDateString()}</td><td class="space-x-2 text-right">{#if document.status === 'ready'}<a class="link" href={documentContentUrl(document.id)} target="_blank" rel="noreferrer">{document.media_type === 'application/pdf' || document.media_type.startsWith('image/') ? 'Preview' : 'Download'}</a>{:else if document.status === 'failed'}<button class="link" on:click={() => retry(document.id)}>Retry</button>{/if}<button class="link text-red-500" on:click={() => removeDocument(document)}>Delete</button></td></tr>{/each}
						{#if documents.length === 0}<tr><td colspan="4" class="py-12 text-center text-sm text-gray-400">No {collection === 'slides' ? 'slides' : 'past exams'} yet. Upload a PDF, PPTX, DOCX, PNG, or JPEG.</td></tr>{/if}
					</tbody></table></div>
				{:else}<div class="flex min-h-[24rem] items-center justify-center text-center"><div><div class="text-5xl">📚</div><h2 class="mt-4 text-xl font-medium text-gray-900 dark:text-white">Choose a course</h2><p class="mt-2 text-sm text-gray-500">Create a course to start its Slides and Past Exams libraries.</p></div></div>{/if}
			</div>
		</section>
	</div>
</div>
