# Guided Learning Interface

## NEW: Easy-to-Use Menu System

Just type `linalg-tutor` and follow the menus - no commands to memorize!

## What Changed

### Before (Command-Based)
```bash
# Had to know specific commands
linalg-tutor exercise practice vectors -n 5
linalg-tutor generate practice vector_add --count 10
linalg-tutor visualize vector 3,4
```

### After (Menu-Based)
```bash
# Just run linalg-tutor and navigate menus
linalg-tutor

# Shows interactive menu:
Main Menu:
  → Continue Learning: Vectors
  📖 Select Chapter
  📊 View Progress
  🎲 Quick Practice (Random)
  ❓ Help & Commands
  ⚙️  Settings
  🚪 Exit
```

## Menu Flow

### 1. Main Menu (First Run)
```
╭────────────────────────────────────────╮
│ Welcome to Linear Algebra Tutor!      │
│                                        │
│ An interactive learning application    │
│ for mastering undergraduate linear     │
│ algebra through practice.              │
╰────────────────────────────────────────╯

Main Menu:
  📖 Select Chapter
  📊 View Progress
  🎲 Quick Practice (Random)
  ❓ Help & Commands
  ⚙️  Settings
  🚪 Exit
```

### 2. Main Menu (After Starting)
```
╭────────────────────────────────────────╮
│ Your Progress                          │
│                                        │
│ Current Chapter: Vectors               │
│ Chapters Completed: 0/10               │
│ Exercises Completed: 15                │
│ Progress: 0%                           │
╰────────────────────────────────────────╯

Main Menu:
  → Continue Learning: Vectors
  📖 Select Chapter
  📊 View Progress
  🎲 Quick Practice (Random)
  ❓ Help & Commands
  ⚙️  Settings
  🚪 Exit
```

### 3. Chapter Selection
```
Learning Path
┏━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Ch.┃ Chapter        ┃ Status       ┃ Description                  ┃
┡━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1  │ Vectors        │ → Current    │ Vector operations            │
│ 2  │ Matrices       │ ○ Not Started│ Matrix operations            │
│ 3  │ Linear Systems │ ○ Not Started│ Solving systems              │
│ 4  │ Vector Spaces  │ ○ Not Started│ Subspaces, basis             │
│ 5  │ Orthogonality  │ ○ Not Started│ Orthogonal projections       │
│ 6  │ Determinants   │ ○ Not Started│ Properties, expansion        │
│ 7  │ Eigenvalues    │ ○ Not Started│ Characteristic equation      │
│ 8  │ Transformations│ ○ Not Started│ Linear transformations       │
│ 9  │ Decompositions │ ○ Not Started│ SVD, QR, LU                  │
│ 10 │ Applications   │ ○ Not Started│ Real-world applications      │
└────┴────────────────┴──────────────┴──────────────────────────────┘

Select a chapter:
  → Chapter 1: Vectors
    Chapter 2: Matrices
    Chapter 3: Linear Systems
    ...
  ← Back to Main Menu
```

### 4. Chapter Menu
```
╭────────────────────────────────────────╮
│ Chapter 1: Vectors                     │
│                                        │
│ Vector operations and properties       │
╰────────────────────────────────────────╯

What would you like to do?
  📚 Practice Curated Exercises
  ∞ Generate Practice Problems
  👁 View Visualizations
  🔧 Advanced Solvers
  ← Back to Chapter Selection
```

### 5. Practice Session
After selecting "Practice Curated Exercises":
```
How many exercises? (1-16, default: 5): 3

Starting practice session: 3 exercises

[Fixed-screen interface appears for each exercise]
```

### 6. Progress View
```
Learning Path
[Same chapter list with status indicators]

Total Exercises Completed       45
Total Time Spent               325.4s
Overall Progress                20%
```

## Features

### Session State
- **Automatically saved** - Your progress is always saved
- **Resume anytime** - Continue where you left off
- **Progress tracking** - See how many chapters completed

### Status Indicators
- `✓ Complete` - Chapter finished
- `⚡ In Progress` - Started but not finished
- `→ Current` - Your current chapter
- `○ Not Started` - Haven't started yet

### Navigation
- **Arrow keys** - Move up/down in menus
- **Enter** - Select option
- **Ctrl+C** - Go back/exit (handled gracefully)
- **No typing** - Everything is menu-driven

### Learning Path
10 ordered chapters:
1. **Vectors** - Start here!
2. **Matrices** - After vectors
3. **Linear Systems** - After matrices
4. **Vector Spaces** - After linear systems
5. **Orthogonality** - After vector spaces
6. **Determinants** - After orthogonality
7. **Eigenvalues** - After determinants
8. **Transformations** - After eigenvalues
9. **Decompositions** - After transformations
10. **Applications** - Final chapter!

### Per-Chapter Options

#### 📚 Practice Curated Exercises
- Hand-crafted exercises with detailed explanations
- Choose how many (1 to total available)
- Fixed-screen interface with hints

#### ∞ Generate Practice Problems
- Infinite randomized exercises
- Choose generator type
- Choose how many

#### 👁 View Visualizations
- See visual demonstrations
- Commands shown for manual use

#### 🔧 Advanced Solvers
- Step-by-step problem solving
- Commands shown for manual use

## Settings

### ⚙️ Settings Menu
```
Settings:
  🔄 Reset All Progress
  📁 View Data Location
  ← Back to Main Menu
```

**Reset All Progress:**
- Clears all saved progress
- Confirmation required
- Cannot be undone

**View Data Location:**
- Shows where progress is stored
- `~/.linalg_tutor/data/session_state.json`
- `~/.linalg_tutor/data/progress.db`

## Quick Practice

**🎲 Quick Practice (Random):**
- Choose number of exercises
- Random mix from all available generators
- Great for review or warm-up

## Old Commands Still Work

All original commands still function:
```bash
linalg-tutor exercise practice vectors
linalg-tutor generate practice vector_add
linalg-tutor visualize vector 3,4
linalg-tutor solve eigenvalues '4,-2;1,1'
linalg-tutor topics
linalg-tutor start
```

But now you don't need to memorize them - just use the menu!

## Example Session

```bash
$ linalg-tutor

Welcome to Linear Algebra Tutor!

Main Menu:
  📖 Select Chapter
  [user selects]

Learning Path shows all 10 chapters

Select a chapter:
  → Chapter 1: Vectors
  [user selects]

What would you like to do?
  📚 Practice Curated Exercises
  [user selects]

How many exercises? (1-16, default: 5): 3

[Practice session with fixed-screen interface]

Session Complete: Vectors

Exercises Completed    3
Correct                3
Accuracy               100%
Total Time            45.2s

[Returns to chapter menu automatically]
```

## Benefits

### For New Users
- ✅ No commands to memorize
- ✅ Guided through chapters
- ✅ Clear progression path
- ✅ Can't get lost

### For Advanced Users
- ✅ Quick navigation with menus
- ✅ Can still use commands if preferred
- ✅ Progress automatically tracked
- ✅ See overall progress at a glance

### For Everyone
- ✅ Your progress is never lost
- ✅ Resume exactly where you stopped
- ✅ Clear visual feedback
- ✅ Graceful Ctrl+C handling throughout

## Data Storage

All progress stored in:
- **Session State**: `~/.linalg_tutor/data/session_state.json`
- **Exercise History**: `~/.linalg_tutor/data/progress.db`

You can:
- Backup these files
- Delete to start fresh
- Transfer between computers

## Tips

1. **Start with Chapter 1** - The chapters build on each other
2. **Complete exercises in each chapter** - Don't rush ahead
3. **Use Quick Practice** - Great for review
4. **Check Progress often** - Stay motivated!
5. **Take your time** - Quality over quantity

## Future Enhancements

Planned features:
- [ ] Unlock chapters only after prerequisites complete
- [ ] Achievement badges for milestones
- [ ] Daily practice streaks
- [ ] Recommended review based on performance
- [ ] Export progress reports

## Get Started

Just type:
```bash
linalg-tutor
```

And start your journey through linear algebra!
