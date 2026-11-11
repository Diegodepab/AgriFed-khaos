/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/agrifed_khaos/templates/**/*.html",
    "./src/agrifed_khaos/static/**/*.js"
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
