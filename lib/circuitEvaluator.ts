import { Gate, Connection } from '@/types/gates';
import { GateLogic } from './gateLogic';

export class CircuitEvaluator {
  static evaluateCircuit(gates: Gate[], connections: Connection[]): Gate[] {
    // Create a copy of gates to avoid mutating the original
    const updatedGates = gates.map(gate => ({ ...gate, inputs: [...gate.inputs] }));

    // Build a map for quick lookup
    const gateMap = new Map(updatedGates.map(gate => [gate.id, gate]));

    // Reset all inputs except INPUT gates
    updatedGates.forEach(gate => {
      if (gate.type !== 'INPUT') {
        gate.inputs = new Array(GateLogic.getInputCount(gate.type)).fill(false);
      }
    });

    // Propagate signals through connections
    connections.forEach(connection => {
      const fromGate = gateMap.get(connection.fromGateId);
      const toGate = gateMap.get(connection.toGateId);

      if (fromGate && toGate) {
        // Calculate the output of the source gate
        fromGate.output = GateLogic.evaluate(fromGate);

        // Set the input of the target gate
        if (connection.toPort < toGate.inputs.length) {
          toGate.inputs[connection.toPort] = fromGate.output;
        }
      }
    });

    // Evaluate all gates multiple times to propagate signals through the circuit
    // This handles multi-level circuits
    for (let iteration = 0; iteration < updatedGates.length; iteration++) {
      connections.forEach(connection => {
        const fromGate = gateMap.get(connection.fromGateId);
        const toGate = gateMap.get(connection.toGateId);

        if (fromGate && toGate) {
          fromGate.output = GateLogic.evaluate(fromGate);
          if (connection.toPort < toGate.inputs.length) {
            toGate.inputs[connection.toPort] = fromGate.output;
          }
        }
      });

      // Evaluate all gates
      updatedGates.forEach(gate => {
        gate.output = GateLogic.evaluate(gate);
      });
    }

    return updatedGates;
  }
}
