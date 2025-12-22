import { Gate, GateType } from '@/types/gates';

export class GateLogic {
  static evaluate(gate: Gate): boolean {
    switch (gate.type) {
      case 'AND':
        return gate.inputs.length >= 2 && gate.inputs.every(input => input);

      case 'OR':
        return gate.inputs.length >= 2 && gate.inputs.some(input => input);

      case 'NOT':
        return gate.inputs.length >= 1 && !gate.inputs[0];

      case 'INPUT':
        return gate.inputs[0] || false;

      case 'OUTPUT':
        return gate.inputs[0] || false;

      default:
        return false;
    }
  }

  static getInputCount(type: GateType): number {
    switch (type) {
      case 'NOT':
      case 'OUTPUT':
        return 1;
      case 'AND':
      case 'OR':
        return 2;
      case 'INPUT':
        return 0;
      default:
        return 0;
    }
  }

  static getOutputCount(type: GateType): number {
    switch (type) {
      case 'OUTPUT':
        return 0;
      default:
        return 1;
    }
  }

  static createGate(type: GateType, position: { x: number; y: number }): Gate {
    const inputCount = this.getInputCount(type);
    return {
      id: `gate-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      position,
      inputs: new Array(inputCount).fill(false),
      output: false,
    };
  }
}
