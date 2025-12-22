'use client';

import { Connection, Gate } from '@/types/gates';

interface ConnectionComponentProps {
  connection: Connection;
  gates: Gate[];
  onDelete?: () => void;
}

export default function ConnectionComponent({ connection, gates, onDelete }: ConnectionComponentProps) {
  const fromGate = gates.find(g => g.id === connection.fromGateId);
  const toGate = gates.find(g => g.id === connection.toGateId);

  if (!fromGate || !toGate) return null;

  // Calculate port positions
  const fromX = fromGate.position.x + 100 + 8; // gate width + half port size
  const fromY = fromGate.position.y + 30; // center of gate

  const inputPortSpacing = 16;
  const toX = toGate.position.x - 8;
  const toY = toGate.position.y + 30 + (connection.toPort * inputPortSpacing);

  // Create a curved path
  const midX = (fromX + toX) / 2;
  const path = `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`;

  const isActive = fromGate.output;

  return (
    <g onClick={onDelete} className="cursor-pointer group">
      <path
        d={path}
        fill="none"
        stroke={isActive ? '#fbbf24' : '#6b7280'}
        strokeWidth="3"
        className="group-hover:stroke-red-500 transition-colors"
      />
      <circle
        cx={(fromX + toX) / 2}
        cy={(fromY + toY) / 2}
        r="6"
        fill="transparent"
        className="group-hover:fill-red-500"
      />
    </g>
  );
}
