# SMU Masters in Applied AI - Course Website

A comprehensive 10-course graduate program website for the Masters in Applied Artificial Intelligence at SMU Lyle School of Engineering.

## Project Structure

```
/
├── index.html                    # Main program homepage
├── assets/
│   ├── styles.css               # Global stylesheet with SMU branding
│   └── quiz.js                  # Interactive quiz functionality
├── resources/
│   ├── syllabus.html            # Full program syllabus
│   ├── curated-resources.html   # External learning resources
│   └── faq.html                 # Frequently asked questions
└── courses/
    ├── course-01-statistics/    # Statistical Inference
    ├── course-02-programming/   # Programming & Computational Thinking
    ├── course-03-data-science/  # Applied Data Science & Machine Learning
    ├── course-04-prompt-engineering/
    ├── course-05-llm-architectures/
    ├── course-06-distributed-systems/
    ├── course-07-business-transformation/
    ├── course-08-autonomous-agents/
    ├── course-09-human-ai-interaction/
    └── course-10-product-incubation/
```

## Each Course Contains

- `index.html` - Course landing page with 15-week module grid
- `syllabus.html` - Detailed course syllabus
- `readings.html` - Recommended readings
- `glossary.html` - Key terms and definitions
- `week-XX/` folders containing:
  - `index.html` - Weekly module page
  - `quiz.html` - Knowledge check
  - `discussion.html` - Discussion topics
  - Code examples and exercises

## Design System

### Brand Colors
- **SMU Blue:** #354CA1
- **SMU Red:** #CC0035
- **Background:** #f7f9fc
- **Card Background:** #ffffff

### Typography
- System font stack for optimal performance
- Line height: 1.6 for readability

## Development

This is a static HTML/CSS/JS site with no build tools required. Simply open `index.html` in a browser to view.

### Local Development

```bash
# Using Python's built-in server
python -m http.server 8000

# Using Node.js serve
npx serve .
```

## License

© 2026 SMU Lyle School of Engineering
