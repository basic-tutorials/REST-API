export type GateType = 'AND' | 'OR' | 'NOT' | 'INPUT' | 'OUTPUT';

export interface Position {
  x: number;
  y: number;
}

export interface Connection {
  id: string;
  fromGateId: string;
  fromPort: number;
  toGateId: string;
  toPort: number;
}

export interface Gate {
  id: string;
  type: GateType;
  position: Position;
  inputs: boolean[];
  output: boolean;
  label?: string;
}

export interface CircuitState {
  gates: Gate[];
  connections: Connection[];
  selectedGateId: string | null;
  isDragging: boolean;
}
