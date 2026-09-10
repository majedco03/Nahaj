<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import { getQuiz, getAttempt, startAttempt, submitAnswer, completeAttempt, type Attempt, type QuizDetail, type QuizQuestion } from '$lib/apis/nahaj';

	let quizId = '';
	let quiz: QuizDetail | null = null;
	let attempt: Attempt | null = null;
	let answers: Record<string, string> = {};
	let optionOrder: Record<string, string[]> = {};
	let index = 0;
	let loading = true;
	let saving = false;
	let completed = false;

	$: currentQuestion = attempt?.questions[index] as QuizQuestion | undefined;
	$: currentOptions = currentQuestion ? (optionOrder[currentQuestion.id] ?? currentQuestion.options.map((option) => option.option_id).filter(Boolean).slice()) : [];

	function shuffle<T>(values: T[]) {
		const result = [...values];
		const random = new Uint32Array(1);
		for (let i = result.length - 1; i > 0; i -= 1) {
			crypto.getRandomValues(random);
			const j = random[0] % (i + 1);
			[result[i], result[j]] = [result[j], result[i]];
		}
		return result;
	}

	function persistOrder() {
		if (!attempt) return;
		window.localStorage.setItem(`nahaj-attempt-order-${attempt.id}`, JSON.stringify(optionOrder));
		window.localStorage.setItem(`nahaj-active-attempt-${quizId}`, attempt.id);
	}

	function restoreOrder(nextAttempt: Attempt) {
		const stored = window.localStorage.getItem(`nahaj-attempt-order-${nextAttempt.id}`);
		try { optionOrder = stored ? JSON.parse(stored) : {}; } catch { optionOrder = {}; }
		for (const question of nextAttempt.questions) {
			const ids = question.options.map((option) => option.option_id).filter(Boolean);
			if (!Array.isArray(optionOrder[question.id]) || optionOrder[question.id].length !== ids.length || optionOrder[question.id].some((id) => !ids.includes(id))) optionOrder[question.id] = shuffle(ids);
		}
		answers = Object.fromEntries(nextAttempt.questions.map((question) => [question.id, question.selected_option_id ?? '']).filter(([, value]) => value));
		persistOrder();
	}

	async function load() {
		quizId = $page.params.id ?? '';
		loading = true;
		try {
			quiz = await getQuiz(quizId);
			const activeId = window.localStorage.getItem(`nahaj-active-attempt-${quizId}`);
			if (activeId) {
				const restored = await getAttempt(activeId).catch(() => null);
				if (restored && restored.status === 'in_progress' && restored.quiz_id === quizId) { attempt = restored; restoreOrder(restored); }
			}
		} catch (err) { toast.error(`${err}`); } finally { loading = false; }
	}

	async function begin() {
		try {
			attempt = await startAttempt(quizId);
			index = 0;
			restoreOrder(attempt);
		} catch (err) { toast.error(`${err}`); }
	}

	async function choose(optionId: string) {
		if (!attempt || !currentQuestion || saving || completed) return;
		saving = true;
		answers = { ...answers, [currentQuestion.id]: optionId };
		try { await submitAnswer(attempt.id, currentQuestion.id, optionId); } catch (err) { toast.error(`${err}`); }
		finally { saving = false; }
	}

	async function finish() {
		if (!attempt) return;
		if (Object.keys(answers).length < attempt.questions.length && !confirm('Some questions are unanswered. Submit anyway?')) return;
		try {
			attempt = await completeAttempt(attempt.id);
			completed = true;
			window.localStorage.removeItem(`nahaj-active-attempt-${quizId}`);
		} catch (err) { toast.error(`${err}`); }
	}

	function optionText(id: string) { return currentQuestion?.options.find((option) => option.option_id === id)?.text ?? id; }
	onMount(load);
</script>

<svelte:head><title>{quiz?.title ?? 'Quiz'} / Nahaj</title></svelte:head>

<div class="min-h-0 min-w-0 flex-1 overflow-y-auto px-4 py-8 @3xl:px-10"><div class="mx-auto max-w-3xl space-y-6">
	{#if loading}<div class="py-20 text-center text-sm text-gray-400">Loading quiz…</div>{:else if !quiz}<div class="empty"><h1 class="text-xl font-medium">Quiz not found</h1><button class="primary mt-4" on:click={() => goto('/quizzes')}>Back to quizzes</button></div>{:else}<div class="flex flex-wrap items-start justify-between gap-4"><div><a class="text-xs text-indigo-600 hover:underline" href="/quizzes">← All quizzes</a><h1 class="mt-2 text-3xl font-semibold text-gray-900 dark:text-white">{quiz.title}</h1><p class="mt-2 text-sm text-gray-500">{quiz.questions.length} questions · {quiz.source_collections.join(', ') || 'course material'}</p></div>{#if !attempt}<button class="primary" on:click={begin}>Start attempt</button>{/if}</div>
		{#if !attempt}<div class="empty"><div class="text-4xl">🧠</div><h2 class="mt-3 text-lg font-medium text-gray-900 dark:text-white">Ready when you are</h2><p class="mt-1 text-sm text-gray-500">Options will be shuffled securely on this device and stay in the same order if you refresh.</p></div>{:else if !completed && attempt.status === 'in_progress'}<div class="flex items-center justify-between text-sm text-gray-500"><span>Question {index + 1} of {attempt.questions.length}</span><span>{Object.keys(answers).length} answered</span></div><div class="progress"><span style={`width:${((index + 1) / attempt.questions.length) * 100}%`}></span></div><div class="question-card"><div class="flex flex-wrap gap-1">{#each attempt.questions as question, questionIndex}<button class="question-dot {questionIndex === index ? 'question-current' : ''} {answers[question.id] ? 'question-answered' : ''}" on:click={() => (index = questionIndex)} aria-label={`Go to question ${questionIndex + 1}`}>{questionIndex + 1}</button>{/each}</div><h2 class="mt-6 text-xl font-medium leading-relaxed text-gray-900 dark:text-white">{currentQuestion?.prompt}</h2><div class="mt-6 space-y-3">{#each currentOptions as optionId, optionIndex}<button class="option {answers[currentQuestion?.id ?? ''] === optionId ? 'option-selected' : ''}" on:click={() => choose(optionId)}><span class="option-letter">{String.fromCharCode(65 + optionIndex)}</span><span>{optionText(optionId)}</span></button>{/each}</div><div class="mt-8 flex justify-between"><button class="secondary" disabled={index === 0} on:click={() => (index -= 1)}>Previous</button>{#if index === attempt.questions.length - 1}<button class="primary" on:click={finish}>Submit quiz</button>{:else}<button class="primary" on:click={() => (index += 1)}>Next</button>{/if}</div></div>{:else}<div class="question-card"><div class="rounded-xl bg-indigo-50 p-4 dark:bg-indigo-950/30"><p class="text-sm text-gray-500">Score</p><p class="mt-1 text-4xl font-semibold text-indigo-600 dark:text-indigo-300">{Math.round(attempt.score_percent ?? 0)}%</p><p class="mt-1 text-sm text-gray-500">Weak topics: {attempt.weak_topics?.length ? attempt.weak_topics.join(', ') : 'none identified'}</p></div><h2 class="mt-7 text-lg font-medium text-gray-900 dark:text-white">Review answers</h2><div class="mt-4 space-y-4">{#each attempt.questions as question, questionIndex}<article class="review"><p class="font-medium text-gray-900 dark:text-white">{questionIndex + 1}. {question.prompt}</p><p class="mt-2 text-sm {question.is_correct ? 'text-emerald-600' : 'text-red-600'}">Your answer: {question.selected_option_id ? question.options.find((option) => option.option_id === question.selected_option_id)?.text : 'Not answered'}</p><p class="mt-1 text-sm text-emerald-600">Correct answer: {question.options.find((option) => option.option_id === question.correct_option_id)?.text}</p>{#if question.explanation}<p class="mt-2 text-sm text-gray-500">{question.explanation}</p>{/if}{#if question.topic}<span class="mt-2 inline-block text-xs text-gray-400">Topic: {question.topic}</span>{/if}</article>{/each}</div><button class="secondary mt-6" on:click={() => goto('/quizzes')}>Back to quizzes</button></div>{/if}
	{/if}
</div></div>
