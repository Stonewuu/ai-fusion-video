export function generationDurationScale(min: number, max: number, options?: readonly number[]) {
  const values = options?.length ? [...options].sort((left, right) => left - right) : undefined;
  return {
    min: values ? 0 : min,
    max: values ? values.length - 1 : max,
    first: values ? values[0] : min,
    last: values ? values[values.length - 1] : max,
    position: (seconds: number) => values ? values.indexOf(seconds) : seconds,
    seconds: (position: number) => values ? values[position] : position,
  };
}
