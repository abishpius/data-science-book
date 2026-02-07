/**
 * AWS Exam Practice Tool - Quiz Logic
 */

let questions = [];
let currentQuestionIndex = 0;
let score = 0;
let userAnswers = {}; // Map of index -> array of selected options
let examStarted = false;
let feedbackShown = false;

// DOM Elements
const startScreen = document.getElementById("start-screen");
const quizScreen = document.getElementById("quiz-screen");
const resultScreen = document.getElementById("result-screen");
const startBtn = document.getElementById("start-btn");
const checkBtn = document.getElementById("check-btn");
const nextBtn = document.getElementById("next-btn");
const prevBtn = document.getElementById("prev-btn");
const finishBtn = document.getElementById("finish-btn");
const restartBtn = document.getElementById("restart-btn");

const questionText = document.getElementById("question-text");
const optionsContainer = document.getElementById("options-container");
const questionImageContainer = document.getElementById(
  "question-image-container",
);
const questionImage = document.getElementById("question-image");
const qNumberText = document.getElementById("q-number");
const multipleTag = document.getElementById("multiple-tag");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const scoreText = document.getElementById("score-text");
const quizStats = document.getElementById("quiz-stats");

/**
 * Initialize the app
 */
async function init() {
  try {
    const response = await fetch("questions.json");
    questions = await response.json();
    console.log(`Loaded ${questions.length} questions.`);

    startBtn.disabled = false;
  } catch (error) {
    console.error("Error loading questions:", error);
    questionText.innerHTML =
      "Error loading questions. Please ensure questions.json exists.";
  }
}

/**
 * Start the exam
 */
function startExam() {
  examStarted = true;
  currentQuestionIndex = 0;
  score = 0;
  userAnswers = {};

  startScreen.style.display = "none";
  quizScreen.style.display = "block";
  quizStats.style.display = "flex";

  showQuestion(currentQuestionIndex);
  updateProgress();
}

/**
 * Show a question by index
 */
function showQuestion(index) {
  const q = questions[index];
  feedbackShown = false;

  // Update header
  qNumberText.innerText = `Question ${index + 1} of ${questions.length}`;
  multipleTag.style.display = q.multiple ? "inline-block" : "none";

  // Update text
  questionText.innerText = q.question;

  // Update image
  if (q.image) {
    questionImage.src = q.image;
    questionImageContainer.style.display = "block";
    // Handle image load error
    questionImage.onerror = () => {
      console.warn(`Image not found: ${q.image}`);
      questionImageContainer.style.display = "none";
    };
  } else {
    questionImageContainer.style.display = "none";
  }

  // Render options
  optionsContainer.innerHTML = "";
  const alphabet = ["A", "B", "C", "D", "E", "F"];

  q.options.forEach((opt, i) => {
    const optionEl = document.createElement("div");
    optionEl.className = "option";
    if (userAnswers[index] && userAnswers[index].includes(i)) {
      optionEl.classList.add("selected");
    }

    optionEl.innerHTML = `
            <div class="option-letter">${alphabet[i]}</div>
            <div class="option-content">${opt}</div>
        `;

    optionEl.onclick = () => selectOption(i);
    optionsContainer.appendChild(optionEl);
  });

  // Update Nav Buttons
  prevBtn.disabled = index === 0;
  nextBtn.disabled = index === questions.length - 1;
  nextBtn.style.display = "block";
  nextBtn.innerText = "Next";
  nextBtn.classList.remove("btn-primary");
  nextBtn.classList.add("btn-secondary");

  checkBtn.style.display = "inline-block";
  checkBtn.disabled = !userAnswers[index] || userAnswers[index].length === 0;
}

/**
 * Handle option selection
 */
function selectOption(optionIndex) {
  if (feedbackShown) return;

  const q = questions[currentQuestionIndex];
  if (!userAnswers[currentQuestionIndex]) {
    userAnswers[currentQuestionIndex] = [];
  }

  const indexInSelected =
    userAnswers[currentQuestionIndex].indexOf(optionIndex);

  if (q.multiple) {
    // Multi-select logic
    if (indexInSelected > -1) {
      userAnswers[currentQuestionIndex].splice(indexInSelected, 1);
    } else {
      userAnswers[currentQuestionIndex].push(optionIndex);
    }
  } else {
    // Single-select logic
    userAnswers[currentQuestionIndex] = [optionIndex];
  }

  // Re-render selection states
  const optionEls = optionsContainer.querySelectorAll(".option");
  optionEls.forEach((el, i) => {
    if (userAnswers[currentQuestionIndex].includes(i)) {
      el.classList.add("selected");
    } else {
      el.classList.remove("selected");
    }
  });

  checkBtn.disabled = userAnswers[currentQuestionIndex].length === 0;
}

/**
 * Check the answer and provide feedback
 */
function checkAnswer() {
  const q = questions[currentQuestionIndex];
  const selected = userAnswers[currentQuestionIndex] || [];
  const correct = q.answer;

  feedbackShown = true;

  let isCorrect = false;
  // Check if the set of selected matches the set of correct
  if (
    selected.length === correct.length &&
    selected.every((val) => correct.includes(val))
  ) {
    isCorrect = true;
    score++;
  }

  // Show feedback colors
  const optionEls = optionsContainer.querySelectorAll(".option");
  optionEls.forEach((el, i) => {
    el.classList.remove("selected");
    if (correct.includes(i)) {
      el.classList.add("correct");
    } else if (selected.includes(i)) {
      el.classList.add("wrong");
    }
  });

  checkBtn.style.display = "none";
  nextBtn.classList.remove("btn-secondary");
  nextBtn.classList.add("btn-primary");
  nextBtn.disabled = false;

  if (currentQuestionIndex === questions.length - 1) {
    nextBtn.innerText = "See Results";
  } else {
    nextBtn.innerText = "Next Question";
  }

  updateScore();
}

/**
 * Navigation
 */
function nextQuestion() {
  if (currentQuestionIndex < questions.length - 1) {
    currentQuestionIndex++;
    showQuestion(currentQuestionIndex);
    updateProgress();
  } else {
    showResults();
  }
}

function prevQuestion() {
  if (currentQuestionIndex > 0) {
    currentQuestionIndex--;
    showQuestion(currentQuestionIndex);
    updateProgress();
  }
}

/**
 * UI Updates
 */
function updateProgress() {
  const progress = ((currentQuestionIndex + 1) / questions.length) * 100;
  progressBar.style.width = `${progress}%`;
  progressText.innerText = `${currentQuestionIndex + 1}/${questions.length}`;
}

function updateScore() {
  // We only count questions where feedback was shown
  const answeredCount = Object.keys(userAnswers).filter((idx) => {
    // Simple heuristic: if we were at this index and moved on, or are at it and feedback shows
    return (
      parseInt(idx) < currentQuestionIndex ||
      (parseInt(idx) === currentQuestionIndex && feedbackShown)
    );
  }).length;

  if (answeredCount > 0) {
    const percentage = Math.round((score / answeredCount) * 100);
    scoreText.innerText = `${percentage}%`;
  }
}

/**
 * Results
 */
function showResults() {
  quizScreen.style.display = "none";
  resultScreen.style.display = "block";
  quizStats.style.display = "none";

  const accuracy = Math.round((score / questions.length) * 100);
  document.getElementById("finished-total").innerText = questions.length;
  document.getElementById("finished-correct").innerText = score;
  document.getElementById("finished-accuracy").innerText = `${accuracy}%`;
  document.getElementById("final-score-text").innerText = `${accuracy}%`;

  // Circular chart animation
  const circle = document.getElementById("score-circle-fill");
  circle.setAttribute("stroke-dasharray", `${accuracy}, 100`);
}

/**
 * Event Listeners
 */
startBtn.onclick = startExam;
checkBtn.onclick = checkAnswer;
nextBtn.onclick = nextQuestion;
prevBtn.onclick = prevQuestion;
finishBtn.onclick = showResults;
restartBtn.onclick = () => {
  resultScreen.style.display = "none";
  startScreen.style.display = "block";
};

// Start initialization
init();
