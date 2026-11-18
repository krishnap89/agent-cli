export interface HttpClient {
  get<T>(url: string): Promise<T>
}

export class FetchClient implements HttpClient {
  async get<T>(url: string): Promise<T> {
    const res = await fetch(url)
    return res.json() as Promise<T>
  }
}

export function createClient(): HttpClient {
  return new FetchClient()
}
