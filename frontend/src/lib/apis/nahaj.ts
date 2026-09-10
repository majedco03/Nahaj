export type Collection = 'slides' | 'past_exam';
export type TaskStatus = 'todo' | 'done' | 'skipped';

export type Semester = {
	id: string;
	name: string;
	start_date: string;
	end_date: string;
	timezone: string;
	active: boolean;
	created_at: string;
	updated_at: string;
};

export type Course = {
	id: string;
	semester_id: string;
	code: string;
	name: string;
	color: string;
	created_at: string;
	updated_at: string;
};

export type DocumentRecord = {
	id: string;
	course_id: string;
	collection: Collection;
	filename: string;
	media_type: string;
	size_bytes: number;
	sha256: string;
	status: 'processing' | 'ready' | 'failed';
	error: string | null;
	created_at: string;
	updated_at: string;
};

export type StudyTask = {
	id: string;
	course_id: string | null;
	title: string;
	notes: string | null;
	start_at: string;
	end_at: string;
	status: TaskStatus;
	origin: 'student' | 'supervisor';
	created_at: string;
	updated_at: string;
};

export type QuizOption = { option_id: string; text: string };
export type QuizQuestion = {
	id: string;
	prompt: string;
	topic: string | null;
	explanation: string | null;
	options: QuizOption[];
};

export type Quiz = {
	id: string;
	course_id: string | null;
	title: string;
	source_collections: string[];
	question_count: number;
	created_at: string;
	latest_score: number | null;
	best_score: number | null;
};

export type QuizDetail = Quiz & { questions: QuizQuestion[] };
export type Attempt = {
	id: string;
	quiz_id: string;
	status: 'in_progress' | 'completed';
	score_percent: number | null;
	weak_topics: string[];
	started_at: string;
	completed_at: string | null;
	questions: Array<QuizQuestion & {
		correct_option_id?: string;
		selected_option_id?: string | null;
		is_correct?: boolean;
	}>;
};

export type Notification = {
	id: string;
	severity: string;
	title: string;
	message: string;
	target_url: string | null;
	read_at: string | null;
	created_at: string;
};

export type Progress = {
	task_completion_percent: number;
	quiz_average_percent: number | null;
	courses: Array<{
		course_id: string;
		code: string;
		name: string;
		completed_tasks: number;
		total_tasks: number;
		quiz_average_percent: number | null;
	}>;
};

type APIError = { error?: { code?: string; message?: string; details?: unknown }; detail?: string };

const BASE = '/api/nahaj/api/v1';

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
	const response = await fetch(`${BASE}${path}`, {
		...init,
		headers: {
			Accept: 'application/json',
			...(init.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
			...(init.headers ?? {})
		}
	});
	if (!response.ok) {
		let body: APIError = {};
		try {
			body = await response.json();
		} catch {
			// Keep the HTTP status when the upstream returned an empty body.
		}
		throw new Error(body.error?.message ?? body.detail ?? `Nahaj request failed (${response.status})`);
	}
	if (response.status === 204) return undefined as T;
	return response.json();
}

export const getSemester = () => request<Semester | null>('/semester');
export const saveSemester = (data: Pick<Semester, 'name' | 'start_date' | 'end_date' | 'timezone'>) =>
	request<Semester>('/semester', { method: 'PUT', body: JSON.stringify(data) });

export const getCourses = () => request<Course[]>('/courses');
export const createCourse = (data: Pick<Course, 'code' | 'name' | 'color'>) =>
	request<Course>('/courses', { method: 'POST', body: JSON.stringify(data) });
export const updateCourse = (id: string, data: Partial<Pick<Course, 'code' | 'name' | 'color'>>) =>
	request<Course>(`/courses/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
export const deleteCourse = (id: string) => request<void>(`/courses/${id}`, { method: 'DELETE' });

export const getDocuments = (courseId: string, collection?: Collection) =>
	request<DocumentRecord[]>(`/courses/${courseId}/documents${collection ? `?collection=${collection}` : ''}`);
export const uploadDocument = (courseId: string, collection: Collection, file: File) => {
	const body = new FormData();
	body.append('collection', collection);
	body.append('file', file);
	return request<DocumentRecord>(`/courses/${courseId}/documents`, { method: 'POST', body });
};
export const getDocument = (id: string) => request<DocumentRecord>(`/documents/${id}`);
export const retryDocument = (id: string) => request<DocumentRecord>(`/documents/${id}/retry`, { method: 'POST' });
export const deleteDocument = (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' });
export const documentContentUrl = (id: string) => `${BASE}/documents/${id}/content`;

export const getPlan = (params: { from?: string; to?: string; course_id?: string; status?: TaskStatus } = {}) => {
	const query = new URLSearchParams();
	Object.entries(params).forEach(([key, value]) => value && query.set(key, value));
	return request<{ semester: Semester | null; tasks: StudyTask[]; summary: { total: number; done: number; completion_percent: number } }>(
		`/plan${query.size ? `?${query.toString()}` : ''}`
	);
};
export const createTask = (data: Omit<StudyTask, 'id' | 'created_at' | 'updated_at' | 'origin'> & { origin?: StudyTask['origin'] }) =>
	request<StudyTask>('/tasks', { method: 'POST', body: JSON.stringify(data) });
export const updateTask = (id: string, data: Partial<StudyTask>) =>
	request<StudyTask>(`/tasks/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
export const deleteTask = (id: string) => request<void>(`/tasks/${id}`, { method: 'DELETE' });

export const getProgress = () => request<Progress>('/progress');

export const getQuizzes = () => request<Quiz[]>('/quizzes');
export const getQuiz = (id: string) => request<QuizDetail>(`/quizzes/${id}`);
export const deleteQuiz = (id: string) => request<void>(`/quizzes/${id}`, { method: 'DELETE' });
export const startAttempt = (quizId: string) =>
	request<Attempt>(`/quizzes/${quizId}/attempts`, { method: 'POST' });
export const getAttempt = (id: string) => request<Attempt>(`/attempts/${id}`);
export const submitAnswer = (attemptId: string, questionId: string, selectedOptionId: string) =>
	request<{ question_id: string; selected_option_id: string; answered: boolean }>(`/attempts/${attemptId}/answers`, {
		method: 'POST',
		body: JSON.stringify({ question_id: questionId, selected_option_id: selectedOptionId })
	});
export const completeAttempt = (id: string) => request<Attempt>(`/attempts/${id}/complete`, { method: 'POST' });

export const getNotifications = (unread = false) =>
	request<Notification[]>(`/notifications${unread ? '?unread=true' : ''}`);
export const markNotificationRead = (id: string, read = true) =>
	request<Notification>(`/notifications/${id}`, { method: 'PATCH', body: JSON.stringify({ read }) });
export const markAllNotificationsRead = () => request<void>('/notifications/read-all', { method: 'POST' });

