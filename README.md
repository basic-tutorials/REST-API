# Logic Gate Builder

A visual circuit builder for creating and testing logic gate circuits using TypeScript and Next.js.

## Features

- Create circuits using AND, OR, and NOT gates
- Visual drag-and-drop interface
- Connect gates with visual wires
- Real-time circuit evaluation
- INPUT and OUTPUT gates for testing
- Interactive gate placement and connections

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## How to Use

1. **Add Gates**: Click the buttons in the toolbar to add different types of gates to the canvas
   - INPUT: Creates an input source (click to toggle on/off)
   - AND: Logic AND gate (outputs true only if all inputs are true)
   - OR: Logic OR gate (outputs true if any input is true)
   - NOT: Logic NOT gate (inverts the input)
   - OUTPUT: Output display (shows the final result)

2. **Move Gates**: Click and drag any gate to reposition it on the canvas

3. **Create Connections**: Click and drag from any port (the small circles on gates) to another port to create a connection
   - Output ports are on the right side of gates
   - Input ports are on the left side of gates
   - Active connections are shown in yellow, inactive in gray

4. **Toggle Inputs**: Click on INPUT gates to toggle their state between true (yellow) and false (gray)

5. **Delete Gates**: Select a gate by clicking on it, then click "Delete Selected"

6. **Clear Circuit**: Click "Clear All" to remove all gates and connections

## Tech Stack

- Next.js 15
- TypeScript
- Tailwind CSS
- React 18

## Project Structure

```
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main page
│   └── globals.css      # Global styles
├── components/
│   ├── CircuitCanvas.tsx      # Main canvas component
│   ├── GateComponent.tsx      # Individual gate rendering
│   └── ConnectionComponent.tsx # Wire rendering
├── lib/
│   ├── gateLogic.ts          # Gate evaluation logic
│   └── circuitEvaluator.ts   # Circuit signal propagation
└── types/
    └── gates.ts              # TypeScript type definitions
```

## Building for Production

```bash
npm run build
npm start
```
