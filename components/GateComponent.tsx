'use client';

import { Gate } from '@/types/gates';
import { GateLogic } from '@/lib/gateLogic';

interface GateComponentProps {
  gate: Gate;
  isSelected: boolean;
  onMouseDown: (e: React.MouseEvent) => void;
  onPortMouseDown: (portIndex: number, isOutput: boolean, e: React.MouseEvent) => void;
  onInputToggle?: (e: React.MouseEvent) => void;
}

export default function GateComponent({
  gate,
  isSelected,
  onMouseDown,
  onPortMouseDown,
  onInputToggle
}: GateComponentProps) {
  const inputCount = GateLogic.getInputCount(gate.type);
  const outputCount = GateLogic.getOutputCount(gate.type);

  const getGateColor = () => {
    switch (gate.type) {
      case 'AND':
        return 'bg-blue-500';
      case 'OR':
        return 'bg-green-500';
      case 'NOT':
        return 'bg-red-500';
      case 'INPUT':
        return gate.inputs[0] ? 'bg-yellow-500' : 'bg-gray-500';
      case 'OUTPUT':
        return gate.output ? 'bg-yellow-500' : 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div
      className={`absolute cursor-move select-none ${isSelected ? 'ring-4 ring-purple-500' : ''}`}
      style={{
        left: gate.position.x,
        top: gate.position.y,
      }}
      onMouseDown={onMouseDown}
    >
      <div className={`relative ${getGateColor()} text-white rounded-lg shadow-lg p-4 min-w-[100px] min-h-[60px] flex items-center justify-center font-bold`}>
        {/* Input ports */}
        {inputCount > 0 && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 flex flex-col gap-2">
            {Array.from({ length: inputCount }).map((_, i) => (
              <div
                key={`input-${i}`}
                className={`w-4 h-4 rounded-full border-2 border-white cursor-pointer hover:scale-125 transition-transform ${
                  gate.inputs[i] ? 'bg-yellow-300' : 'bg-gray-700'
                }`}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onPortMouseDown(i, false, e);
                }}
                title={`Input ${i + 1}`}
              />
            ))}
          </div>
        )}

        {/* Gate label */}
        <div
          className="text-sm"
          onClick={gate.type === 'INPUT' && onInputToggle ? (e) => {
            e.stopPropagation();
            onInputToggle(e);
          } : undefined}
        >
          {gate.type}
          {gate.type === 'INPUT' && <div className="text-xs">(click to toggle)</div>}
        </div>

        {/* Output port */}
        {outputCount > 0 && (
          <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2">
            <div
              className={`w-4 h-4 rounded-full border-2 border-white cursor-pointer hover:scale-125 transition-transform ${
                gate.output ? 'bg-yellow-300' : 'bg-gray-700'
              }`}
              onMouseDown={(e) => {
                e.stopPropagation();
                onPortMouseDown(0, true, e);
              }}
              title="Output"
            />
          </div>
        )}
      </div>
    </div>
  );
}
