<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getNotifications } from '$lib/apis/nahaj';
	import { nahajUnreadNotifications } from '$lib/stores/nahaj';

	let seen = new Set<string>();

	async function refresh(showToasts = true) {
		try {
			const notifications = await getNotifications(true);
			nahajUnreadNotifications.set(notifications.length);
			if (showToasts) {
				for (const notification of notifications.filter((item) => !seen.has(item.id))) {
					toast(notification.title, { description: notification.message });
				}
			}
			seen = new Set(notifications.map((item) => item.id));
		} catch {
			// The API may be starting with the UI; retry on the next interval.
		}
	}

	onMount(() => {
		refresh(false);
		const timer = window.setInterval(() => refresh(true), 30000);
		const onFocus = () => refresh(true);
		window.addEventListener('focus', onFocus);
		return () => {
			window.clearInterval(timer);
			window.removeEventListener('focus', onFocus);
		};
	});
</script>
