# Manual Testing Guide

## To test the new UX, run these commands in a real terminal:

### 1. Quick Demo (Non-interactive)
```bash
source .venv/bin/activate
linalg-tutor demo
```
**Expected:** Shows demo exercise with solution, no crash

### 2. Start Screen
```bash
linalg-tutor start
```
**Expected:** Shows welcome screen with all command examples

### 3. Simple Practice Session (2 exercises)
```bash
linalg-tutor exercise practice vectors -n 2
```

**Test flow:**
1. First exercise appears in fixed-screen layout
2. Try "Get a hint" - hint should appear and stay visible
3. Try "Submit answer" - question should show during input
4. Enter correct answer: [expected answer from question]
5. See ✓ Correct message
6. Second exercise appears
7. Try "Show solution" - shows solution, continues
8. See session summary at end

**What to verify:**
- ✅ Screen doesn't scroll endlessly
- ✅ Question always visible at top
- ✅ Header shows "Exercise 1/2", then "Exercise 2/2"
- ✅ Hints accumulate in status area
- ✅ Question appears when entering answer
- ✅ Summary shows at end

### 4. Generator Practice (3 exercises)
```bash
linalg-tutor generate practice vector_add -n 3
```

**Test flow:**
1. Complete first exercise correctly
2. On second exercise, try "Visualize" - should show ASCII art
3. Press any key to return to main screen
4. Complete the exercise
5. On third exercise, press Ctrl+C
6. Verify: "Session interrupted" message, progress saved, shows summary

**What to verify:**
- ✅ Random exercises generated
- ✅ Visualization works (shows, returns to main screen)
- ✅ Ctrl+C handled gracefully
- ✅ Summary shows completed exercises (1 correct, 1 correct, 1 interrupted)

### 5. Visualization Commands
```bash
linalg-tutor visualize vector 3,4
linalg-tutor visualize matrix '1,2;3,4'
linalg-tutor visualize dot-product 1,2 3,4
```

**Expected:** Each shows visualization, no crash

### 6. Advanced Solver Commands
```bash
linalg-tutor solve gaussian '2,1,-1;-3,-1,2;-2,1,2' -b '8,-11,-3'
linalg-tutor solve eigenvalues '4,-2;1,1'
```

**Expected:** Shows step-by-step solutions, no crash

### 7. List Commands
```bash
linalg-tutor topics
linalg-tutor exercise list vectors
linalg-tutor generate list-generators
```

**Expected:** Shows tables of available content

## Known Issues (OK to have)

- Warnings about deprecated Pydantic/SQLAlchemy (not breaking)
- Piped input doesn't work (intentional - needs TTY)
- Very long questions might overflow panel (rare)

## Success Criteria

✅ No crashes in any command
✅ Fixed-screen layout works (no endless scroll)
✅ Question visible during answer input
✅ Hints accumulate and stay visible
✅ Visualizations show and return to main screen
✅ Ctrl+C saves progress and shows summary
✅ Session summaries appear after exercises
✅ All 31 pytest tests pass
