<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import { deleteQuiz, getCourses, getQuizzes, type Course, type Quiz } from '$lib/apis/nahaj';

	let quizzes: Quiz[] = [];
	let courses: Course[] = [];
	let loading = true;
	let showNew = false;
	let form = { course_id: '', collection: 'slides', count: 10 };
	$: courseById = new Map(courses.map((course) => [course.id, course]));

	function openPrompt() {
		const course = courseById.get(form.course_id);
		const source = form.collection === 'past_exam' ? 'Past Exams' : 'Slides';
		window.localStorage.setItem('nahaj-chat-draft', `Create a ${form.count}-question MCQ quiz for ${course ? `${course.code} (${course.name})` : 'my selected course'} using ${source}. Use exactly four stable option IDs per question, include one correct answer and explanations.`);
		showNew = false;
		goto('/');
	}

	async function load() {
		loading = true;
		try { [quizzes, courses] = await Promise.all([getQuizzes(), getCourses()]); } catch (err) { toast.error(`${err}`); } finally { loading = false; }
	}

	async function remove(quiz: Quiz) {
		if (!confirm(`Delete “${quiz.title}”?`)) return;
		try { await deleteQuiz(quiz.id); quizzes = quizzes.filter((item) => item.id !== quiz.id); } catch (err) { toast.error(`${err}`); }
	}

	onMount(load);
</script>

<svelte:head><title>Quizzes / Nahaj</title></svelte:head>

<div class="min-h-0 min-w-0 flex-1 overflow-y-auto px-4 py-8 @3xl:px-10"><div class="mx-auto max-w-6xl space-y-8">
	<div class="flex flex-wrap items-start justify-between gap-4"><div><p class="text-xs font-medium uppercase tracking-[0.22em] text-indigo-500">Nahaj</p><h1 class="mt-2 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">Quizzes</h1><p class="mt-2 text-sm text-gray-500 dark:text-gray-400">Practice with supervisor-generated MCQs. Every question has four stable options and is graded by option ID.</p></div><button class="primary" on:click={() => (showNew = true)}>New Quiz</button></div>
	{#if loading}<div class="py-20 text-center text-sm text-gray-400">Loading quizzes…</div>{:else if quizzes.length === 0}<div class="empty"><div class="text-4xl">✍️</div><h2 class="mt-3 text-lg font-medium text-gray-900 dark:text-white">No quizzes yet</h2><p class="mt-1 text-sm text-gray-500">Choose a course and source to ask the supervisor for your first quiz.</p><button class="primary mt-5" on:click={() => (showNew = true)}>Create a quiz</button></div>{:else}<div class="grid gap-4 @3xl:grid-cols-2">{#each quizzes as quiz (quiz.id)}<article class="quiz-card"><div class="flex items-start justify-between gap-3"><div><span class="course-tag" style={`--course-color:${courseById.get(quiz.course_id ?? '')?.color ?? '#6366f1'}`}>{courseById.get(quiz.course_id ?? '')?.code ?? 'All courses'}</span><h2 class="mt-3 text-lg font-medium text-gray-900 dark:text-white">{quiz.title}</h2></div><button class="text-xs text-red-500 hover:underline" on:click={() => remove(quiz)}>Delete</button></div><div class="mt-4 flex items-center justify-between text-sm text-gray-500"><span>{quiz.question_count} questions · {quiz.source_collections.join(', ') || 'course material'}</span><span>{new Date(quiz.created_at).toLocaleDateString()}</span></div><div class="mt-5 flex items-center justify-between border-t border-gray-100 pt-4 dark:border-gray-800"><div class="text-xs text-gray-500">Latest: <b class="text-gray-800 dark:text-gray-200">{quiz.latest_score == null ? '—' : `${Math.round(quiz.latest_score)}%`}</b> · Best: <b class="text-gray-800 dark:text-gray-200">{quiz.best_score == null ? '—' : `${Math.round(quiz.best_score)}%`}</b></div><a class="primary" href={`/quizzes/${quiz.id}`}>Open quiz</a></div></article>{/each}</div>{/if}
</div></div>

{#if showNew}<div class="overlay" role="presentation" on:click={(event) => event.target === event.currentTarget && (showNew = false)}><form class="modal" on:submit|preventDefault={openPrompt}><div class="flex items-center justify-between"><h2 class="text-lg font-medium text-gray-900 dark:text-white">New quiz</h2><button type="button" class="text-gray-400" on:click={() => (showNew = false)}>✕</button></div><p class="text-sm text-gray-500">The request will open in the main chat so the supervisor remains the single coordinator.</p><label>Course<select class="field mt-1 w-full" bind:value={form.course_id}><option value="">All courses</option>{#each courses as course}<option value={course.id}>{course.code} · {course.name}</option>{/each}</select></label><label>Source collection<select class="field mt-1 w-full" bind:value={form.collection}><option value="slides">Slides</option><option value="past_exam">Past Exams</option></select></label><label>Question count<input class="field mt-1 w-full" type="number" min="1" max="50" bind:value={form.count} /></label><div class="flex justify-end gap-2"><button type="button" class="secondary" on:click={() => (showNew = false)}>Cancel</button><button class="primary">Open chat</button></div></form></div>{/if}
