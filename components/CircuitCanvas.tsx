'use client';

import { useState, useEffect, useRef } from 'react';
import { Gate, Connection, GateType } from '@/types/gates';
import { GateLogic } from '@/lib/gateLogic';
import { CircuitEvaluator } from '@/lib/circuitEvaluator';
import GateComponent from './GateComponent';
import ConnectionComponent from './ConnectionComponent';

export default function CircuitCanvas() {
  const [gates, setGates] = useState<Gate[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedGateId, setSelectedGateId] = useState<string | null>(null);
  const [draggingGateId, setDraggingGateId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connectingFrom, setConnectingFrom] = useState<{
    gateId: string;
    portIndex: number;
    isOutput: boolean;
  } | null>(null);
  const [tempConnection, setTempConnection] = useState<{ x: number; y: number } | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);

  // Evaluate circuit whenever gates or connections change
  useEffect(() => {
    const evaluatedGates = CircuitEvaluator.evaluateCircuit(gates, connections);
    setGates(evaluatedGates);
  }, [connections]);

  const addGate = (type: GateType) => {
    const newGate = GateLogic.createGate(type, {
      x: Math.random() * 400 + 50,
      y: Math.random() * 300 + 50,
    });
    setGates([...gates, newGate]);
  };

  const handleGateMouseDown = (gateId: string, e: React.MouseEvent) => {
    e.preventDefault();
    const gate = gates.find(g => g.id === gateId);
    if (!gate) return;

    setSelectedGateId(gateId);
    setDraggingGateId(gateId);
    setDragOffset({
      x: e.clientX - gate.position.x,
      y: e.clientY - gate.position.y,
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggingGateId) {
      const newX = e.clientX - dragOffset.x;
      const newY = e.clientY - dragOffset.y;

      setGates(gates.map(gate =>
        gate.id === draggingGateId
          ? { ...gate, position: { x: newX, y: newY } }
          : gate
      ));
    }

    if (connectingFrom && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      setTempConnection({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    }
  };

  const handleMouseUp = () => {
    setDraggingGateId(null);
    setConnectingFrom(null);
    setTempConnection(null);
  };

  const handlePortMouseDown = (
    gateId: string,
    portIndex: number,
    isOutput: boolean,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();

    if (connectingFrom) {
      // Complete the connection
      if (connectingFrom.isOutput && !isOutput) {
        // From output to input
        const newConnection: Connection = {
          id: `conn-${Date.now()}`,
          fromGateId: connectingFrom.gateId,
          fromPort: connectingFrom.portIndex,
          toGateId: gateId,
          toPort: portIndex,
        };
        setConnections([...connections, newConnection]);
      } else if (!connectingFrom.isOutput && isOutput) {
        // From input to output
        const newConnection: Connection = {
          id: `conn-${Date.now()}`,
          fromGateId: gateId,
          fromPort: portIndex,
          toGateId: connectingFrom.gateId,
          toPort: connectingFrom.portIndex,
        };
        setConnections([...connections, newConnection]);
      }
      setConnectingFrom(null);
      setTempConnection(null);
    } else {
      // Start a new connection
      setConnectingFrom({ gateId, portIndex, isOutput });
    }
  };

  const handleInputToggle = (gateId: string) => {
    setGates(gates.map(gate => {
      if (gate.id === gateId && gate.type === 'INPUT') {
        return {
          ...gate,
          inputs: [!gate.inputs[0]],
        };
      }
      return gate;
    }));
  };

  const deleteSelectedGate = () => {
    if (!selectedGateId) return;

    setGates(gates.filter(g => g.id !== selectedGateId));
    setConnections(connections.filter(
      c => c.fromGateId !== selectedGateId && c.toGateId !== selectedGateId
    ));
    setSelectedGateId(null);
  };

  const deleteConnection = (connectionId: string) => {
    setConnections(connections.filter(c => c.id !== connectionId));
  };

  const clearCircuit = () => {
    setGates([]);
    setConnections([]);
    setSelectedGateId(null);
  };

  // Get temp connection path
  const getTempConnectionPath = () => {
    if (!connectingFrom || !tempConnection) return null;

    const fromGate = gates.find(g => g.id === connectingFrom.gateId);
    if (!fromGate) return null;

    let fromX: number, fromY: number;

    if (connectingFrom.isOutput) {
      fromX = fromGate.position.x + 100 + 8;
      fromY = fromGate.position.y + 30;
    } else {
      const inputPortSpacing = 16;
      fromX = fromGate.position.x - 8;
      fromY = fromGate.position.y + 30 + (connectingFrom.portIndex * inputPortSpacing);
    }

    const midX = (fromX + tempConnection.x) / 2;
    return `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${tempConnection.y}, ${tempConnection.x} ${tempConnection.y}`;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Toolbar */}
      <div className="bg-gray-800 p-4 flex gap-2 flex-wrap shadow-lg">
        <button
          onClick={() => addGate('INPUT')}
          className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition-colors font-semibold"
        >
          + INPUT
        </button>
        <button
          onClick={() => addGate('AND')}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors font-semibold"
        >
          + AND
        </button>
        <button
          onClick={() => addGate('OR')}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors font-semibold"
        >
          + OR
        </button>
        <button
          onClick={() => addGate('NOT')}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors font-semibold"
        >
          + NOT
        </button>
        <button
          onClick={() => addGate('OUTPUT')}
          className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors font-semibold"
        >
          + OUTPUT
        </button>

        <div className="flex-1" />

        <button
          onClick={deleteSelectedGate}
          disabled={!selectedGateId}
          className="px-4 py-2 bg-red-700 text-white rounded hover:bg-red-800 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors font-semibold"
        >
          Delete Selected
        </button>
        <button
          onClick={clearCircuit}
          className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors font-semibold"
        >
          Clear All
        </button>
      </div>

      {/* Canvas */}
      <div
        ref={canvasRef}
        className="flex-1 relative overflow-hidden bg-gray-900"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={() => setSelectedGateId(null)}
      >
        {/* SVG for connections */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {connections.map(connection => (
            <ConnectionComponent
              key={connection.id}
              connection={connection}
              gates={gates}
              onDelete={() => deleteConnection(connection.id)}
            />
          ))}

          {/* Temporary connection while dragging */}
          {tempConnection && getTempConnectionPath() && (
            <path
              d={getTempConnectionPath()!}
              fill="none"
              stroke="#9ca3af"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          )}
        </svg>

        {/* Gates */}
        {gates.map(gate => (
          <GateComponent
            key={gate.id}
            gate={gate}
            isSelected={selectedGateId === gate.id}
            onMouseDown={(e) => handleGateMouseDown(gate.id, e)}
            onPortMouseDown={(portIndex, isOutput, e) =>
              handlePortMouseDown(gate.id, portIndex, isOutput, e)
            }
            onInputToggle={gate.type === 'INPUT' ? () => handleInputToggle(gate.id) : undefined}
          />
        ))}

        {/* Instructions */}
        {gates.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-gray-500 text-center">
              <h2 className="text-2xl font-bold mb-4">Logic Gate Builder</h2>
              <p className="mb-2">Click the buttons above to add gates to the canvas</p>
              <p className="mb-2">Drag gates to move them</p>
              <p className="mb-2">Click and drag from ports to create connections</p>
              <p>Click on INPUT gates to toggle their value</p>
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="bg-gray-800 p-2 text-gray-300 text-sm">
        Gates: {gates.length} | Connections: {connections.length}
        {selectedGateId && ` | Selected: ${gates.find(g => g.id === selectedGateId)?.type}`}
      </div>
    </div>
  );
}
