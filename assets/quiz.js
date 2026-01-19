/**
 * SMU Applied AI - Interactive Quiz System
 * Provides immediate feedback for multiple choice questions
 */

document.addEventListener('DOMContentLoaded', function() {
  initializeQuizzes();
});

function initializeQuizzes() {
  const quizQuestions = document.querySelectorAll('.quiz-question');
  
  quizQuestions.forEach((question, index) => {
    const options = question.querySelectorAll('.quiz-option');
    const correctAnswer = question.dataset.correct;
    const explanations = JSON.parse(question.dataset.explanations || '{}');
    
    options.forEach((option) => {
      option.addEventListener('click', function() {
        // Prevent re-answering
        if (question.classList.contains('answered')) return;
        
        const selectedValue = this.dataset.value;
        const isCorrect = selectedValue === correctAnswer;
        
        // Mark question as answered
        question.classList.add('answered');
        
        // Remove selected state from all options
        options.forEach(opt => opt.classList.remove('selected'));
        
        // Mark correct/incorrect
        options.forEach(opt => {
          if (opt.dataset.value === correctAnswer) {
            opt.classList.add('correct');
          } else if (opt === this && !isCorrect) {
            opt.classList.add('incorrect');
          }
        });
        
        // Show explanation
        let explanationDiv = question.querySelector('.quiz-explanation');
        if (!explanationDiv) {
          explanationDiv = document.createElement('div');
          explanationDiv.className = 'quiz-explanation';
          question.appendChild(explanationDiv);
        }
        
        const explanation = explanations[selectedValue] || 'No explanation available.';
        const status = isCorrect ? 
          '<strong style="color: #28a745;">✓ Correct!</strong>' : 
          '<strong style="color: #CC0035;">✗ Incorrect</strong>';
        
        explanationDiv.innerHTML = `${status}<p>${explanation}</p>`;
        explanationDiv.classList.add('show');
      });
    });
  });
  
  // Initialize reset buttons
  const resetButtons = document.querySelectorAll('.quiz-reset');
  resetButtons.forEach(button => {
    button.addEventListener('click', resetQuiz);
  });
}

function resetQuiz() {
  const quizQuestions = document.querySelectorAll('.quiz-question');
  
  quizQuestions.forEach(question => {
    // Remove answered state
    question.classList.remove('answered');
    
    // Reset all options
    const options = question.querySelectorAll('.quiz-option');
    options.forEach(opt => {
      opt.classList.remove('selected', 'correct', 'incorrect');
    });
    
    // Hide explanations
    const explanation = question.querySelector('.quiz-explanation');
    if (explanation) {
      explanation.classList.remove('show');
    }
  });
}

/**
 * Calculate and display quiz score
 */
function calculateScore() {
  const questions = document.querySelectorAll('.quiz-question');
  let correct = 0;
  let answered = 0;
  
  questions.forEach(question => {
    if (question.classList.contains('answered')) {
      answered++;
      const correctAnswer = question.dataset.correct;
      const selectedOption = question.querySelector('.quiz-option.correct, .quiz-option.incorrect');
      if (selectedOption && selectedOption.classList.contains('correct') && 
          selectedOption.dataset.value === correctAnswer) {
        correct++;
      }
    }
  });
  
  return {
    correct: correct,
    total: questions.length,
    answered: answered,
    percentage: questions.length > 0 ? Math.round((correct / questions.length) * 100) : 0
  };
}

/**
 * Display score summary
 */
function showScore() {
  const score = calculateScore();
  const scoreDisplay = document.getElementById('quiz-score');
  
  if (scoreDisplay) {
    scoreDisplay.innerHTML = `
      <div class="resources-section">
        <h3>Your Score</h3>
        <p><strong>${score.correct}</strong> out of <strong>${score.total}</strong> correct (${score.percentage}%)</p>
        <p>${score.answered === score.total ? 'Quiz complete!' : `${score.total - score.answered} questions remaining.`}</p>
      </div>
    `;
  }
}
