export class Logger {
  info(msg: string): void {}

  invalidate(): void {}
}

export namespace Log {
  export function tag(s: string): string {
    return `[${s}]`
  }

  export function line(s: string): string {
    return tag(s) + s
  }
}
