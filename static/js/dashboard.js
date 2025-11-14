// static/js/dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // Auto-refresh every 5 minutes
    setInterval(() => {
        location.reload();
    }, 5 * 60 * 1000);

    // Add floating animation to stat cards
    const cards = document.querySelectorAll('.stat-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('animate__animated', 'animate__fadeInUp');
    });

    // Add glow to map on hover
    const map = document.querySelector('.map-container');
    if (map) {
        map.addEventListener('mouseenter', () => {
            map.style.boxShadow = '0 0 30px rgba(0, 255, 255, 0.6)';
        });
        map.addEventListener('mouseleave', () => {
            map.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.3)';
        });
    }
});